/**
 * Lets a display formula that carries several equations on one line reflow.
 *
 * Theory pages routinely state two or three related equations in a single
 * display block, separated by the author's own \quad or \qquad, because on a
 * wide screen they belong together. KaTeX lays that row out as one unbreakable
 * line, so on a phone the reader has to scroll the formula sideways to reach
 * the last equation, and the block is centred, which means the first equation
 * starts off-screen too.
 *
 * KaTeX already chops the row into unbreakable chunks (it breaks after every
 * top-level relation and binary operator, as the TeXbook prescribes), but those
 * are not equation boundaries: letting them wrap would strand an "=" at the end
 * of a line. So the split is made at the author's own widest top-level spacer
 * instead, which is exactly where one equation ends and the next begins, and
 * each equation becomes a flex item. The separator is then re-expressed as the
 * row's column gap, and a column gap is only drawn *between* items on the same
 * line: a wide viewport gets the identical single centred row it got before,
 * and a narrow one stacks one equation per line, each centred, with no dangling
 * 2 em space hanging off the line ends.
 *
 * Nothing here depends on the page, the language or the theme, and the source
 * markdown keeps stating the equations the way they read best: one line.
 */
export function rehypeWrappableMath() {
  const classesOf = (node) =>
    node?.type === 'element' && Array.isArray(node.properties?.className)
      ? node.properties.className
      : [];
  const hasClass = (node, name) => classesOf(node).includes(name);
  // KaTeX renamed its layout classes in 0.18 (`base` -> `katex-base`,
  // `strut` -> `katex-strut`) to stop colliding with page styles. rehype-katex
  // still renders with the 0.16 line, so both spellings are accepted rather
  // than tying this pass to whichever one is installed.
  const isChunk = (node) => hasClass(node, 'katex-base') || hasClass(node, 'base');
  const isStrut = (node) => hasClass(node, 'katex-strut') || hasClass(node, 'strut');
  // KaTeX writes horizontal space as a right margin on an empty span.
  const spacerEm = (node) => {
    const match = /margin-right:\s*([\d.]+)em/.exec(node.properties?.style ?? '');
    return match ? Number(match[1]) : 0;
  };
  // The struts carry the chunk's height and depth, so every piece a chunk is
  // split into needs its own copy.
  const cloneStrut = (strut) =>
    strut && { ...strut, properties: { ...strut.properties } };

  return (tree) => {
    (function visit(node) {
      if (!hasClass(node, 'katex-display')) {
        node.children?.forEach(visit);
        return;
      }
      const katex = node.children?.find((c) => hasClass(c, 'katex'));
      const html = katex?.children?.find((c) => hasClass(c, 'katex-html'));
      const chunks = html?.children ?? [];
      // A tagged or explicitly line-broken formula puts a tag or a newline
      // among the chunks and already controls its own layout.
      if (!chunks.length || !chunks.every(isChunk)) return;

      // The widest top-level spacer is the author's own outermost separator:
      // in `a = b \quad (b > 0), \qquad c = d` the \quad ties the condition to
      // its equation and only the \qquad separates equations.
      let gap = 0;
      for (const chunk of chunks) {
        for (const child of chunk.children ?? []) {
          if (hasClass(child, 'mspace')) gap = Math.max(gap, spacerEm(child));
        }
      }
      if (gap < 1) return; // No \quad-sized separator: a single equation.

      const parts = [];
      let part = [];
      for (const chunk of chunks) {
        const children = chunk.children ?? [];
        // KaTeX always opens a chunk with its strut, but do not assume it: a
        // first child taken for a strut it is not would be copied into every
        // piece the chunk is split into.
        const strut = isStrut(children[0]) ? children[0] : null;
        const rest = strut ? children.slice(1) : children;
        const piece = [];
        const emit = () => {
          if (!piece.length) return;
          part.push({
            ...chunk,
            properties: { ...chunk.properties },
            children: [cloneStrut(strut), ...piece].filter(Boolean),
          });
          piece.length = 0;
        };
        for (const child of rest) {
          // Only the separator itself is taken out, and the column gap puts it
          // back. Everything else stays exactly where it was, including the
          // thin inter-atom space KaTeX writes next to it, so the row keeps its
          // original width to the pixel while it fits on one line.
          if (hasClass(child, 'mspace') && spacerEm(child) >= gap) {
            emit();
            if (part.length) {
              parts.push(part);
              part = [];
            }
            continue;
          }
          piece.push(child);
        }
        emit();
      }
      if (part.length) parts.push(part);
      if (parts.length < 2) return;

      html.children = parts.map((children) => ({
        type: 'element',
        tagName: 'span',
        properties: { className: ['katex-eq-part'] },
        children,
      }));
      html.properties = {
        ...html.properties,
        className: [...classesOf(html), 'katex-eq-row'],
        style: `${html.properties?.style ? `${html.properties.style};` : ''}--katex-eq-gap:${gap}em`,
      };
    })(tree);
  };
}
