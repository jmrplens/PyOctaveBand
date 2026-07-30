/**
 * The site's navigation tree, in one place.
 *
 * astro.config.mjs hands this array to Starlight and scripts/check-sidebar.mjs
 * reads it to work out what the rendered tree has to look like, so the checks
 * follow the content: adding a guide or a whole group is one edit here and the
 * audit expects it on the next run, with nothing to keep in step by hand.
 */
import { apiSidebar } from '../generated/api-sidebar.mjs';

// Stock Starlight sidebar. Every group is `collapsed: true`, and
// Starlight forces open the chain of groups that holds the current page,
// so a reader always lands with their own branch unfolded and everything
// else closed. A group label is plain text inside a `<summary>` and
// cannot be a link in stock Starlight (the sidebar schema declares
// `attrs: z.never()` on groups and gives them no `slug` or `link`), so
// each group that has a landing page carries it as an explicit
// `Overview` entry, first in the group. The API reference is its own
// top-level group, next to Reference rather than inside it.
//
// `collapsed` is a server-rendered boolean, so it cannot differ between
// desktop and phone, and the earlier "expanded on desktop, folded on
// mobile" idea is superseded: the tree arrives folded on both. Undoing
// that is a one-word edit, `collapsed: false` on the groups concerned.
export const sidebar = [
  {
    // The only group without an Overview row, on purpose: "Getting started"
    // is already its front door, so an overview page here would only restate
    // the row below it.
    label: 'Start',
    translations: { es: 'Inicio' },
    collapsed: true,
    items: [
      'getting-started',
      // The map of the nine areas. A sibling entry point rather than a parent:
      // the area overviews stay the breadcrumb ancestors of their own guides.
      { slug: 'guides', label: 'All guides', translations: { es: 'Todas las guías' } },
      'reference/why-phonometry',
      'about',
    ],
  },
  {
    label: 'Core signal analysis',
    translations: { es: 'Análisis de señal' },
    collapsed: true,
    items: [
      { slug: 'guides/sections/core-signal-analysis', label: 'Overview', translations: { es: 'Resumen' } },
      'guides/sound-level-meter',
      {
        label: 'Octave filtering',
        translations: { es: 'Filtrado en octavas' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/octave-filtering', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/filter-banks',
          'guides/filter-gallery',
          'guides/filter-compliance',
          'guides/block-processing',
          'guides/multichannel',
        ],
      },
      {
        label: 'Levels and weighting',
        translations: { es: 'Niveles y ponderación' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/levels-weighting', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/weighting',
          'guides/special-weightings',
          'guides/time-weighting',
          'guides/levels',
          'guides/environmental-levels',
          'guides/spanish-noise-regulation',
        ],
      },
      {
        label: 'Signals and spectra',
        translations: { es: 'Señales y espectros' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/signals-spectra', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/spectral-analysis',
          'guides/miso-coherence',
          'guides/time-frequency',
          'guides/cepstrum-echoes',
          'guides/synchronous-averaging',
          'guides/machine-diagnostics',
          'guides/correlation-delay',
          'guides/test-signals',
          'guides/system-measurement',
        ],
      },
      {
        label: 'Calibration and uncertainty',
        translations: { es: 'Calibración e incertidumbre' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/calibration-uncertainty', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/calibration',
          'guides/gum-uncertainty',
          'guides/data-qualification',
        ],
      },
    ],
  },
  {
    label: 'Hearing and perception',
    translations: { es: 'Audición y percepción' },
    collapsed: true,
    items: [
      { slug: 'guides/sections/hearing-perception', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Psychoacoustics',
        translations: { es: 'Psicoacústica' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/psychoacoustics', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/loudness',
          'guides/advanced-loudness',
          'guides/sound-quality',
          'guides/tone-prominence',
          'guides/tone-audibility',
          'guides/psychoacoustic-annoyance',
        ],
      },
      {
        label: 'Speech',
        translations: { es: 'Habla' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/speech', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/speech-transmission',
          'guides/speech-intelligibility',
          'guides/objective-intelligibility',
        ],
      },
      {
        label: 'Hearing and exposure',
        translations: { es: 'Audición y exposición' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/hearing-exposure', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/hearing-threshold',
          'guides/noise-induced-hearing-loss',
          'guides/occupational-exposure',
        ],
      },
    ],
  },
  {
    label: 'Rooms and buildings',
    translations: { es: 'Salas y edificación' },
    collapsed: true,
    items: [
      { slug: 'guides/sections/rooms-buildings', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Room acoustics',
        translations: { es: 'Acústica de salas' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/room-acoustics', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/room-impulse-response',
          'guides/room-acoustics',
          'guides/open-plan-acoustics',
          'guides/room-image-sources',
          'guides/room-noise',
          'guides/reverberation-prediction',
          'guides/enclosed-space-absorption',
        ],
      },
      {
        label: 'Sound insulation',
        translations: { es: 'Aislamiento acústico' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/sound-insulation', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/insulation-field',
          'guides/insulation-lab',
          'guides/insulation-intensity',
          'guides/insulation-survey',
          'guides/flanking-lab',
          'guides/heavy-impact-sources',
          'guides/insulation-ratings',
          'guides/facade-insulation',
          'guides/spanish-building-code',
        ],
      },
      {
        label: 'Insulation design',
        translations: { es: 'Diseño del aislamiento' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/insulation-design', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/insulation-prediction',
          'guides/detailed-prediction',
          'guides/panel-sound-insulation',
          'guides/impact-improvement',
          'guides/dynamic-stiffness',
        ],
      },
    ],
  },
  {
    label: 'Materials and surfaces',
    translations: { es: 'Materiales y superficies' },
    collapsed: true,
    items: [
      { slug: 'guides/sections/materials-surfaces', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Absorbers',
        translations: { es: 'Absorbentes' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/absorbers', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/absorption-measurement',
          'guides/airflow-resistance',
          'guides/impedance-tube',
          'guides/porous-absorbers',
          'guides/metamaterial-absorbers',
        ],
      },
      {
        label: 'Diffusers and surfaces',
        translations: { es: 'Difusores y superficies' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/diffusion-surfaces', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/diffusers',
          'guides/metadiffusers',
          'guides/road-absorption',
        ],
      },
    ],
  },
  {
    label: 'Vibration and structure-borne sound',
    translations: { es: 'Vibración y ruido estructural' },
    collapsed: true,
    items: [
      { slug: 'guides/sections/vibration', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Structure-borne sources',
        translations: { es: 'Fuentes de ruido estructural' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/structure-borne', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/mechanical-mobility',
          'guides/junction-transmission',
          'guides/transfer-stiffness',
          'guides/vibration-sound-power',
          'guides/structure-borne-power',
          'guides/installed-structure-borne',
        ],
      },
      {
        label: 'Human vibration',
        translations: { es: 'Vibración en humanos' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/human-vibration', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/human-vibration',
          'guides/multiple-shock-vibration',
        ],
      },
    ],
  },
  {
    label: 'Environment and transport',
    translations: { es: 'Medio ambiente y transporte' },
    collapsed: true,
    items: [
      { slug: 'guides/sections/environment-transport', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Outdoor sound',
        translations: { es: 'Sonido en exteriores' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/outdoor-sound', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/outdoor-propagation',
          'guides/cnossos-rail-emission',
          'guides/cnossos-road-emission',
          'guides/ground-barriers',
          'guides/atmospheric-refraction',
          'guides/impulse-prominence',
        ],
      },
      {
        label: 'Aircraft and wind energy',
        translations: { es: 'Aeronaves y energía eólica' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/aircraft-wind', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/aircraft-noise',
          'guides/airport-noise',
          'guides/rotorcraft-noise',
          'guides/wind-turbine-noise',
        ],
      },
    ],
  },
  {
    label: 'Underwater acoustics',
    translations: { es: 'Acústica submarina' },
    collapsed: true,
    items: [
      { slug: 'guides/sections/underwater', label: 'Overview', translations: { es: 'Resumen' } },
      'guides/underwater-acoustics',
      'guides/underwater-propagation',
      'guides/underwater-solvers',
      'guides/marine-mammal-exposure',
    ],
  },
  {
    label: 'Sources and devices',
    translations: { es: 'Fuentes y dispositivos' },
    collapsed: true,
    items: [
      { slug: 'guides/sections/sources-devices', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Sound power and intensity',
        translations: { es: 'Potencia acústica e intensidad' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/sound-power', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/intensity',
          'guides/sound-power',
          'guides/sound-power-pressure',
          'guides/sound-power-reverberation',
          'guides/sound-power-intensity',
        ],
      },
      {
        label: 'Electroacoustics',
        translations: { es: 'Electroacústica' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/electroacoustics', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/electroacoustics',
          'guides/loudspeakers',
          'guides/microphones',
          'guides/swept-sine-distortion',
          'guides/program-loudness',
        ],
      },
      {
        label: 'Noise control',
        translations: { es: 'Control de ruido' },
        collapsed: true,
        items: [
          { slug: 'guides/sections/noise-control', label: 'Overview', translations: { es: 'Resumen' } },
          'guides/silencers',
          'guides/duct-path',
          'guides/room-to-room',
          'guides/noise-control',
        ],
      },
    ],
  },
  {
    label: 'Wave simulation',
    translations: { es: 'Simulación de ondas' },
    collapsed: true,
    items: [
      { slug: 'guides/sections/simulation', label: 'Overview', translations: { es: 'Resumen' } },
      'guides/fdtd-simulation',
      'guides/elastic-waves',
    ],
  },
  {
    label: 'Reference',
    translations: { es: 'Referencia' },
    collapsed: true,
    items: [
      { slug: 'reference', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Theory',
        translations: { es: 'Teoría' },
        collapsed: true,
        items: [
          { slug: 'reference/theory', label: 'Overview', translations: { es: 'Resumen' } },
          'reference/theory/signal-analysis',
          'reference/theory/perception',
          'reference/theory/rooms-buildings',
          'reference/theory/materials-surfaces',
          'reference/theory/environment-transport',
          'reference/theory/vibration',
        ],
      },
      'reference/conformance',
      // Next to the conformance report on purpose: the two pages are the same
      // evidence story seen from both sides, what the library computes against
      // the standard, and where the printed standard is the thing that is
      // wrong. They cross-link each other.
      {
        slug: 'reference/errata',
        label: 'Errata in published sources',
        translations: { es: 'Erratas de las fuentes publicadas' },
      },
      'reference/bibliography',
      'reference/glossary',
    ],
  },
  // The generated API tree, kept as its own top-level group so the
  // machine-written half of the site is separated from the written one.
  apiSidebar,
];
