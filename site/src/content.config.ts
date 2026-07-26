import { defineCollection, z } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

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

export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({
      extend: z.object({
        references: references.optional(),
      }),
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
