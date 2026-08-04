/**
 * The site's own icons, drawn for it.
 *
 * Starlight's set is a closed list of about thirty general-purpose glyphs plus
 * the Seti file-type icons, and an acoustics library needs none of them: the
 * best a topic could do there was a database for the ocean and a puzzle piece
 * for absorbers, which say nothing and are read as the thing they actually
 * depict. These say what the topic is: two buildings, a porous layer over its
 * backing, a body shaking between two arcs, a wavefront on the mesh it is
 * solved on.
 *
 * HOW TO DRAW ANOTHER ONE
 *
 * - 24 by 24 grid, the same one Starlight uses, so a new icon sits at the same
 *   optical size as the carets and the social marks around it. Keep the
 *   drawing inside 2.5 to 21.5 on both axes.
 * - Strokes, not silhouettes, and no `stroke-width` of your own: the renderer
 *   (src/components/PhIcon.astro) supplies 1.7 with round caps and joins,
 *   which is what keeps drawings by different hands looking like one set. A
 *   solid shape needs `fill="currentColor" stroke="none"` on its element; the
 *   aircraft and the small dots are the only ones that use it.
 * - One bold shape and at most one accent. Everything here was drawn twice
 *   because the first version was legible at 48 px and a smudge at the 16 px
 *   it actually renders at.
 * - Check what it is read as, not what it depicts. Wavefronts under a surface
 *   drew a smiling face; a hydrophone on its cable drew a pair of horns; an
 *   ear with a ring inside it was a map pin. `pnpm run icons:sheet` renders
 *   every icon at both sizes and in a list in both themes, which is where all
 *   three were caught.
 *
 * The sheet it writes, src/assets/icon-sheet.svg, is committed: it is the
 * index of the set, and a pull request that changes a drawing shows the change
 * in it.
 */

/**
 * Every topic in src/data/topics.mjs has one, keyed by topic id. Rendered in
 * the three places a topic names itself: the list in the sidebar, the coverage
 * table on the landing page, and the heading of the topic's own overview.
 */
export const topicIcons = {
  // A flag planted: the beginning of the trail, not a rocket, which every
  // documentation site uses and which says "launch" rather than "start here".
  start: `
    <path d="M6 21V4.2" />
    <path d="M6 5.2h10.2l-2.7 3.6 2.7 3.6H6z" />
  `,

  // A level trace: quiet, one loud event, the decay. Two features rather than
  // five, because at 16 px a busy trace closes into a smudge.
  signal: `
    <path d="M2.6 12.6h3.2l2.6-8.4 3.4 15.6 2.8-9.8 2 4.2h4.8" />
  `,

  // An ear: the outer helix with the canal curling into it. The canal is a curl
  // rather than a ring, because a ring inside a teardrop is a map pin.
  perception: `
    <path d="M6.4 9.8a5.6 5.6 0 0 1 11.2 0c0 2.8-1.9 3.9-3.1 5.2-1 1.1-1.2 2.2-1.7 3.2-.5 1-1.4 1.5-2.5 1.5a2.5 2.5 0 0 1-2.5-2.5" />
    <path d="M11.6 13.8c0-1.3.7-2 1.4-2.7a2.4 2.4 0 1 0-4-1.8" />
  `,

  // Two buildings side by side, low and tall. The topic is rooms and the walls
  // between them, and a facade says that at any size.
  buildings: `
    <path d="M2.8 20.4V11h7.4v9.4" />
    <path d="M10.2 20.4V4.2h10.6v16.2" />
    <path d="M2 20.4h20" />
    <path d="M13.6 8.4h3.8M13.6 12.4h3.8M5.6 15h1.8" />
  `,

  // A porous layer over its backing: the absorber and the wall behind it, which
  // is the pair every measurement in the topic is of.
  materials: `
    <rect x="3" y="5.4" width="18" height="7.6" rx="1.3" />
    <rect x="3" y="15.4" width="18" height="4.2" rx="1.3" />
    <g fill="currentColor" stroke="none">
      <circle cx="7.6" cy="9.2" r="1.15" />
      <circle cx="12" cy="9.2" r="1.15" />
      <circle cx="16.4" cy="9.2" r="1.15" />
    </g>
  `,

  // A body shaking between two arcs: one bold shape and one accent, which is
  // all a 16 px drawing of an oscillation can carry.
  vibration: `
    <rect x="8.4" y="6.6" width="7.2" height="10.8" rx="1.4" />
    <path d="M5 9.2a5.2 5.2 0 0 0 0 5.6" />
    <path d="M19 9.2a5.2 5.2 0 0 1 0 5.6" />
  `,

  // A car. The topic is road and rail noise and the maps drawn from them, and
  // the silhouette survives being small and blurred, which a road does not.
  environment: `
    <path d="M3.4 16.6v-2.4l1.9-4.5A1.8 1.8 0 0 1 7 8.5h10a1.8 1.8 0 0 1 1.7 1.2l1.9 4.5v2.4z" />
    <path d="M4.4 13.4h15.2" />
    <circle cx="7.4" cy="16.6" r="1.7" />
    <circle cx="16.6" cy="16.6" r="1.7" />
  `,

  // An aircraft climbing away, with the ground track under it: a flyover,
  // which is the event every certification and contour method measures.
  aircraft: `
    <path d="M20.6 3.4c-.8-.8-2-.6-3 .4l-3.3 3.3-8.1-2.3-1.5 1.5 6.6 4-2.8 2.8-3-.4-1.1 1.1 2.9 1.8 1.8 2.9 1.1-1.1-.4-3 2.8-2.8 4 6.6 1.5-1.5-2.3-8.1 3.3-3.3c1-1 1.2-2.2.4-3z" fill="currentColor" stroke="none" />
    <path d="M3 21h9" />
  `,

  // A ship radiating down through the surface it floats on. The topic is
  // propagation in water, ship radiated noise and what it does to the animals
  // living in it, and the vessel names the source rather than the fauna. The
  // abstractions all failed first: wavefronts under a surface drew a face, a
  // hydrophone on its cable drew a pair of horns, and the waveguide, which is
  // the figure the topic actually turns on, closed into a blob.
  underwater: `
    <path d="M2.2 9.6h19.6" />
    <path d="M4.8 5.2h14.4l-2.6 4.2H7.4z" />
    <path d="M12 2.2v3" />
    <path d="M7.2 14a5.9 5.9 0 0 0 9.6 0" />
    <path d="M4 18.4a10.6 10.6 0 0 0 16 0" />
  `,

  // A measurement microphone on its stand: the instrument the topic is about,
  // and the one every source in it is measured with.
  devices: `
    <rect x="9" y="2.8" width="6" height="10.4" rx="3" />
    <path d="M5.8 11.2a6.2 6.2 0 0 0 12.4 0" />
    <path d="M12 17.4v3.8" />
    <path d="M9 21.2h6" />
  `,

  // A wavefront on the mesh it is solved on: the site's own mark is that same
  // figure, a grid with curves crossing it.
  simulation: `
    <rect x="3" y="3" width="18" height="18" rx="1.6" />
    <path d="M9 3v18M15 3v18M3 9h18M3 15h18" opacity="0.4" />
    <circle cx="12" cy="12" r="5.2" />
    <circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none" />
  `,

  // An open book: the standards, the theory and the glossary the rest of the
  // site cites.
  reference: `
    <path d="M12 6.6C10.4 5.1 8 4.6 4 4.6v12.8c4 0 6.4.5 8 2 1.6-1.5 4-2 8-2V4.6c-4 0-6.4.5-8 2z" />
    <path d="M12 6.6v14.8" />
  `,

  // Brackets and a slash: the generated reference, which is the code itself.
  api: `
    <path d="M8.6 7.4 4.2 12l4.4 4.6" />
    <path d="M15.4 7.4 19.8 12l-4.4 4.6" />
    <path d="M13.6 5.4 10.4 18.6" />
  `,
};
