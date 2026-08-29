import { defineCollection, z } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';
import { topicSchema } from 'starlight-sidebar-topics/schema';

// Typed bibliography declared in each page's frontmatter and rendered as a
// single APA-7 "References" section by src/components/References.astro (wired
// through the MarkdownContent override). Every entry type shares the base
// fields; the discriminated union adds the type-specific ones. `doi` is the
// bare identifier (e.g. `10.1121/1.1915637`), never a URL: the component
// builds the https://doi.org/ link from it.
const referenceBase = {
  /** Work title, sentence case. Rendered in italics except for articles. */
  title: z.string().min(1),
  /** Publication year. A string admits "n.d." and ranges such as "2016-2017". */
  year: z.union([z.number().int(), z.string().min(1)]).optional(),
  /** Canonical landing page (publisher catalogue, store record, project page). */
  url: z.string().url().optional(),
  /** Bare DOI, e.g. "10.1201/9781315372150". */
  doi: z
    .string()
    .regex(/^10\.\S+$/, 'doi must be the bare identifier, e.g. "10.1121/1.1915637"')
    .optional(),
  /** One or two sentences tying the work to this page (what it anchors here). */
  note: z.string().optional(),
  /**
   * Marks a load-bearing entry: the standard the page actually implements, or
   * the work its method is named after. The page header chips run
   * (src/components/PageChips.astro) shows a capped number of entries, and
   * marked ones are shown first, so a page with a long bibliography still
   * advertises the right sources. Everything else keeps frontmatter order.
   * Optional: a page that marks nothing simply falls back to that order.
   */
  primary: z.boolean().optional(),
};

/** Pre-formatted APA names, one per author: "Surname, I. I.". */
const authorList = z.array(z.string().min(1)).nonempty();

const reference = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('standard'),
    /** Issuing body, spelled out: "International Organization for Standardization". */
    organization: z.string().min(1),
    /** Designation with edition/amendment: "ISO 354:2003", "ICAO Doc 9501, 3rd ed.". */
    designation: z.string().min(1),
    /** Only when it differs from the issuing organization. */
    publisher: z.string().min(1).optional(),
    ...referenceBase,
  }),
  z.object({
    type: z.literal('book'),
    authors: authorList,
    /** Localized edition string: "3rd ed." / "3.ª ed.". */
    edition: z.string().min(1).optional(),
    publisher: z.string().min(1),
    ...referenceBase,
  }),
  z.object({
    type: z.literal('article'),
    authors: authorList,
    journal: z.string().min(1),
    volume: z.union([z.number().int(), z.string().min(1)]).optional(),
    issue: z.union([z.number().int(), z.string().min(1)]).optional(),
    /** Page range, e.g. "82-108", or an article number such as "050005". */
    pages: z.string().min(1).optional(),
    ...referenceBase,
  }),
  z.object({
    type: z.literal('web'),
    /** Personal authors, or use `organization` for corporate authorship. */
    authors: authorList.optional(),
    organization: z.string().min(1).optional(),
    /** Site name shown after the title: "GitHub", "ECAC". */
    siteName: z.string().min(1).optional(),
    ...referenceBase,
  }),
  z.object({
    type: z.literal('report'),
    authors: authorList.optional(),
    organization: z.string().min(1).optional(),
    /** Report designation: "ECAC.CEAC Doc 29, 4th ed., Vol. 2". */
    number: z.string().min(1).optional(),
    /** Publishing institution, when it differs from the authoring organization. */
    institution: z.string().min(1).optional(),
    ...referenceBase,
  }),
]);

const references = z
  .array(reference)
  .superRefine((refs, ctx) => {
    for (const [i, ref] of refs.entries()) {
      if (
        (ref.type === 'web' || ref.type === 'report') &&
        !ref.authors?.length &&
        !ref.organization
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: [i],
          message: `references[${i}] (${ref.type}) needs "authors" or "organization"`,
        });
      }
    }
  });

// The conformance artefact, `docs/conformance.json`, validated on its way into
// the build. It is generated and committed by `make conformance`, and the site
// only ever reads it -- generating it here would couple every documentation
// build to a 45-second run of the full scientific stack, and would let the site
// publish numbers no reviewer had seen. Declaring it as a collection is what
// turns a malformed field into a build error that names the field's path
// instead of a stack trace inside a component.
const conformanceSide = z.object({
  /** One number. Never set together with `record`. */
  value: z.number().optional(),
  /** Prose shown instead of the formatted number, when a figure is not the answer. */
  label: z.string().optional(),
  /** Named values compared exactly, e.g. `{ Rw: 52, C: -1, Ctr: -5 }`. */
  record: z.record(z.string(), z.number()).optional(),
});

const conformanceCheck = z
  .object({
    /** `domain/citation/quantity`, slugged. The join key and the row anchor. */
    id: z.string().min(1),
    domain: z.string().min(1),
    reference: z.object({
      // Same vocabulary as the page bibliographies above, plus `derivation`
      // for a check whose reference is a closed form and not a document.
      kind: z.enum(['standard', 'book', 'article', 'report', 'web', 'derivation']),
      designation: z.string().min(1),
      /** A string, so it can hold "2014", "2e", "2020 + AMD1:2023", "(2010)". */
      edition: z.string().min(1).optional(),
      clause: z.string().min(1).optional(),
      /** The citation exactly as the check registered it. */
      cite: z.string().min(1),
    }),
    quantity: z.string().min(1),
    /** Dotted path to the symbol under test. Not populated yet. */
    implements: z.string().min(1).optional(),
    kind: z.enum(['scalar', 'mask', 'record', 'count']),
    expected: conformanceSide,
    computed: conformanceSide,
    unit: z.string().min(1).optional(),
    tolerance: z
      .object({
        mode: z.enum(['absolute', 'relative', 'mask']),
        value: z.number().optional(),
      })
      .optional(),
    deviation: z.object({
      value: z.number().optional(),
      label: z.string().optional(),
    }),
    binding: z
      .object({
        frequency_hz: z.number().optional(),
        lower: z.number().optional(),
        upper: z.number().optional(),
      })
      .optional(),
    precision: z.number().int().min(0),
    verdict: z.enum(['pass', 'fail', 'by-design', 'not-applicable']),
  })
  .superRefine((check, ctx) => {
    for (const side of ['expected', 'computed'] as const) {
      if (check[side].value !== undefined && check[side].record !== undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: [side],
          message: `${check.id}: a side is one number or a set of named ones, not both`,
        });
      }
    }
  });

const conformanceSchema = z.object({
  /** Bumped only on a shape change a reader cannot ignore. */
  schema: z.literal(1),
  library: z.string().min(1),
  generator: z.string().min(1),
  counts: z.object({
    checks: z.number().int(),
    passing: z.number().int(),
    failing: z.number().int(),
    domains: z.number().int(),
    /** Citation groups, the figure the project publishes. */
    standards: z.number().int(),
    citations: z.number().int(),
    /** Distinct normative documents, from the split citation. */
    designations: z.number().int(),
    /** Distinct further cited works. */
    sources: z.number().int(),
  }),
  /** The closed unit vocabulary every row's `unit` must come from. */
  units: z.array(z.string().min(1)),
  domains: z.array(
    z.object({
      id: z.string().min(1),
      title: z.string().min(1),
      checks: z.number().int(),
      passing: z.number().int(),
    }),
  ),
  panels: z.array(
    z.object({
      id: z.string().min(1),
      title: z.string().min(1),
      unit: z.string().min(1).optional(),
      rows: z.array(z.record(z.string(), z.unknown())),
    }),
  ),
  checks: z.array(conformanceCheck),
});

export const collections = {
  conformance: defineCollection({
    loader: async () => {
      const { report } = await import('./data/conformance-stats.mjs');
      return [{ id: 'report', ...report }];
    },
    schema: conformanceSchema,
  }),
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({
      // `topic` is the topics plugin's escape hatch: a page that no topic
      // lists names its own. Nothing uses it today, because every page is
      // listed and the plugin fails the build on one that is not, but the
      // frontmatter has to accept it or the escape hatch would be a silent
      // no-op the day it is reached for.
      extend: z.object({
        references: references.optional(),
      }).merge(topicSchema),
    }),
  }),
  i18n: defineCollection({
    loader: i18nLoader(),
    schema: i18nSchema({
      extend: z.object({
        'phonometry.references.title': z.string().optional(),
        'phonometry.report.download': z.string().optional(),
        'phonometry.video.download': z.string().optional(),
        'phonometry.video.fallback': z.string().optional(),
      }),
    }),
  }),
};
