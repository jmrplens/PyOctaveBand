/**
 * The page bibliography, as linked structured data.
 *
 * Every guide already declares a typed `references` list in its frontmatter,
 * and the site renders it as an APA-7 section and as the header chips. None of
 * it reached the structured data, which is a strange gap for a library whose
 * whole claim is that each metric comes from the text of a named standard: the
 * machine-readable layer said "a technical article by one author" and nothing
 * about the 129 designations the corpus implements.
 *
 * The design point that makes this a graph rather than a few hundred loose
 * blobs: each work gets a stable `@id` minted from its designation and hung off
 * `/reference/bibliography/`, which is where the site already aggregates them.
 * The bibliography page emits the full node, every other page emits a
 * reference to that `@id`, so "IEC 61672-1:2013" is one entity cited by many
 * pages rather than many identical strings.
 */

/** Slugify a designation or title into an `@id` fragment. */
function slug(value) {
  return String(value)
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

/**
 * Stable fragment identifier for one reference entry.
 *
 * Standards key on their designation, which is globally unique and stable
 * across editions ("IEC 61672-1:2013"). Everything else keys on type, first
 * author or organization, and year, which is what an APA short cite uses and
 * is stable enough for the same work cited from several pages.
 */
export function referenceId(ref, bibliographyUrl) {
  if (ref.type === 'standard') return `${bibliographyUrl}#std-${slug(ref.designation)}`;
  const who = ref.authors?.[0] ?? ref.organization ?? ref.title;
  const surname = String(who).split(',')[0];
  return `${bibliographyUrl}#${ref.type}-${slug(surname)}-${slug(ref.year ?? 'nd')}`;
}

/** APA-formatted author strings to schema.org Person nodes. */
function people(authors) {
  return (authors ?? []).map((name) => ({ '@type': 'Person', name }));
}

/**
 * Full schema.org node for one reference.
 *
 * There is no schema.org type for a technical standard. `CreativeWork` with an
 * `identifier`, a publisher `Organization` and `genre: "Technical standard"`
 * is the modelling the standards bodies use on their own catalogue pages, so
 * that is what is emitted here.
 */
export function referenceNode(ref, bibliographyUrl) {
  const id = referenceId(ref, bibliographyUrl);
  const common = {
    '@id': id,
    name: ref.title,
    ...(ref.year ? { datePublished: String(ref.year) } : {}),
    ...(ref.url ? { url: ref.url } : {}),
    ...(ref.doi
      ? {
          identifier: { '@type': 'PropertyValue', propertyID: 'DOI', value: ref.doi },
          sameAs: `https://doi.org/${ref.doi}`,
        }
      : {}),
    ...(ref.note ? { description: ref.note } : {}),
  };

  switch (ref.type) {
    case 'standard':
      return {
        ...common,
        '@type': 'CreativeWork',
        alternateName: ref.designation,
        // The DOI branch above would otherwise win; a standard's designation is
        // the identifier that matters, and standards rarely carry a DOI.
        identifier: ref.designation,
        genre: 'Technical standard',
        publisher: { '@type': 'Organization', name: ref.publisher ?? ref.organization },
      };
    case 'book':
      return {
        ...common,
        '@type': 'Book',
        author: people(ref.authors),
        ...(ref.edition ? { bookEdition: ref.edition } : {}),
        publisher: { '@type': 'Organization', name: ref.publisher },
      };
    case 'article':
      return {
        ...common,
        '@type': 'ScholarlyArticle',
        author: people(ref.authors),
        isPartOf: { '@type': 'Periodical', name: ref.journal },
        ...(ref.volume ? { volumeNumber: String(ref.volume) } : {}),
        ...(ref.issue ? { issueNumber: String(ref.issue) } : {}),
        ...(ref.pages ? { pagination: ref.pages } : {}),
      };
    case 'report':
      return {
        ...common,
        '@type': 'Report',
        ...(ref.authors ? { author: people(ref.authors) } : {}),
        ...(ref.number ? { reportNumber: ref.number } : {}),
        publisher: {
          '@type': 'Organization',
          name: ref.institution ?? ref.organization,
        },
      };
    default:
      return {
        ...common,
        '@type': 'WebPage',
        ...(ref.authors ? { author: people(ref.authors) } : {}),
        ...(ref.organization ? { publisher: { '@type': 'Organization', name: ref.organization } } : {}),
      };
  }
}

/**
 * The citation properties for one page.
 *
 * The three are deliberately different claims, and the frontmatter already
 * carries the flag that separates them:
 *   - `about`      what the page is about: the software plus the standards it
 *                  actually implements (the `primary` entries).
 *   - `isBasedOn`  the same primary standards, stated as provenance.
 *   - `citation`   the full bibliography, primary or not.
 *
 * @param {object} options
 * @param {Array}  options.references     Frontmatter `references`, possibly undefined.
 * @param {string} options.bibliographyUrl Absolute URL of the bibliography page.
 * @param {boolean} options.full           Emit whole nodes rather than `@id` stubs.
 */
export function citationsFor({ references, bibliographyUrl, full = false }) {
  const refs = references ?? [];
  if (refs.length === 0) return {};

  const node = (ref) =>
    full ? referenceNode(ref, bibliographyUrl) : { '@id': referenceId(ref, bibliographyUrl) };

  const primary = refs.filter((ref) => ref.primary);
  // A page that marks nothing still has a governing source; fall back to the
  // standards it lists, and to the first entry if it lists none.
  const governing = primary.length
    ? primary
    : refs.filter((ref) => ref.type === 'standard').slice(0, 3);

  return {
    citation: refs.map(node),
    ...(governing.length ? { isBasedOn: governing.map(node) } : {}),
    ...(governing.length ? { governingWorks: governing.map(node) } : {}),
  };
}
