/**
 * Spanish accessible names for the image-zoom controls.
 *
 * starlight-image-zoom writes the zoom button straight into the markdown tree
 * with `aria-label="Zoom image: <alt>"` hard-coded in English, and it exposes
 * no translation hook: the label never passes through Starlight's `i18n:setup`
 * or `Astro.locals.t`. On a bilingual site that leaves every Spanish figure
 * with a control whose accessible name mixes an English verb with a Spanish
 * alt text, which is a WCAG 3.1.2 (Language of Parts) defect and not something
 * an automated audit can see, since the name is present and non-empty.
 *
 * The button only exists after the plugin's own rehype pass has run, so the
 * fix has to run after it. That is the whole reason this ships as a Starlight
 * plugin rather than an entry in `markdown.rehypePlugins`: Starlight adds each
 * plugin's integrations in declaration order, so placing this one after
 * `starlightImageZoom()` puts this transform last in the rehype chain, exactly
 * where the plugin itself sits relative to the site's own passes.
 *
 * The rewrite asserts rather than pattern-matches loosely: a Spanish page that
 * carries a zoom control the plugin labelled some other way fails the build.
 * A silent no-op here would look identical to a page with no figures, and the
 * regression would only ever be noticed by a screen-reader user.
 */

const CONTROL_CLASS = 'starlight-image-zoom-control';
const EN_PREFIX = 'Zoom image';
const ES_PREFIX = 'Ampliar imagen';

/** Reads an attribute that hast may hold under either spelling. */
const attr = (properties, dashed, camel) => properties?.[dashed] ?? properties?.[camel];

/** True for the Spanish half of the content collection. */
function isSpanish(file) {
  const path = (file?.history?.[0] ?? file?.path ?? '').replaceAll('\\', '/');
  return /\/content\/docs\/es(\/|$)/.test(path);
}

export function rehypeZoomLabelsEs() {
  return function transformer(tree, file) {
    if (!isSpanish(file)) return;

    (function visit(node) {
      if (node.type === 'element' && node.tagName === 'button') {
        const classes = String(attr(node.properties, 'class', 'className') ?? '');
        if (classes.split(/\s+/).includes(CONTROL_CLASS)) {
          const key = 'aria-label' in node.properties ? 'aria-label' : 'ariaLabel';
          const label = String(node.properties[key] ?? '');
          if (!label.startsWith(EN_PREFIX)) {
            throw new Error(
              `[zoom-labels] ${file?.path ?? 'unknown page'}: expected an image-zoom ` +
                `control labelled "${EN_PREFIX}...", found "${label}". ` +
                'starlight-image-zoom changed its accessible name; update ' +
                'src/lib/zoom-labels.mjs so the Spanish pages stay translated.',
            );
          }
          node.properties[key] = ES_PREFIX + label.slice(EN_PREFIX.length);
        }
      }
      node.children?.forEach(visit);
    })(tree);
  };
}

/**
 * Starlight plugin wrapper. Declare it after `starlightImageZoom()` so its
 * integration, and therefore its rehype pass, is registered after the one it
 * has to correct.
 */
export function starlightZoomLabels() {
  return {
    name: 'phonometry-zoom-labels',
    hooks: {
      'config:setup'({ addIntegration }) {
        addIntegration({
          name: 'phonometry-zoom-labels-integration',
          hooks: {
            'astro:config:setup': ({ config }) => {
              const processor = config.markdown.processor;
              if (!Array.isArray(processor?.options?.rehypePlugins)) {
                throw new Error(
                  '[zoom-labels] the configured markdown processor exposes no ' +
                    'rehypePlugins array; the Spanish zoom labels cannot be applied.',
                );
              }
              processor.options.rehypePlugins.push(rehypeZoomLabelsEs);
            },
          },
        });
      },
    },
  };
}
