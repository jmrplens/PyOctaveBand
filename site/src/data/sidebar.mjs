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
      'start/getting-started',
      // The map of the ten topics. A sibling entry point rather than a parent:
      // the area overviews stay the breadcrumb ancestors of their own guides.
      { slug: 'start/guides', label: 'All guides', translations: { es: 'Todas las guías' } },
      'start/why-phonometry',
      'start/about',
    ],
  },
  {
    label: 'Core signal analysis',
    translations: { es: 'Análisis de señal' },
    collapsed: true,
    items: [
      { slug: 'signal', label: 'Overview', translations: { es: 'Resumen' } },
      'signal/sound-level-meter',
      {
        label: 'Octave filtering',
        translations: { es: 'Filtrado en octavas' },
        collapsed: true,
        items: [
          { slug: 'signal/filters', label: 'Overview', translations: { es: 'Resumen' } },
          'signal/filters/filter-banks',
          'signal/filters/filter-gallery',
          'signal/filters/filter-compliance',
          'signal/filters/block-processing',
          'signal/filters/multichannel',
        ],
      },
      {
        label: 'Levels and weighting',
        translations: { es: 'Niveles y ponderación' },
        collapsed: true,
        items: [
          { slug: 'signal/levels', label: 'Overview', translations: { es: 'Resumen' } },
          'signal/levels/weighting',
          'signal/levels/special-weightings',
          'signal/levels/time-weighting',
          'signal/levels/levels',
                ],
      },
      {
        label: 'Signals and spectra',
        translations: { es: 'Señales y espectros' },
        collapsed: true,
        items: [
          { slug: 'signal/spectra', label: 'Overview', translations: { es: 'Resumen' } },
          'signal/spectra/spectral-analysis',
          'signal/spectra/miso-coherence',
          'signal/spectra/time-frequency',
          'signal/spectra/cepstrum-echoes',
          'signal/spectra/synchronous-averaging',
              'signal/spectra/correlation-delay',
          'signal/spectra/test-signals',
          'signal/spectra/system-measurement',
        ],
      },
      {
        label: 'Calibration and uncertainty',
        translations: { es: 'Calibración e incertidumbre' },
        collapsed: true,
        items: [
          { slug: 'signal/metrology', label: 'Overview', translations: { es: 'Resumen' } },
          'signal/metrology/calibration',
          'signal/metrology/gum-uncertainty',
          'signal/metrology/data-qualification',
        ],
      },
    ],
  },
  {
    label: 'Hearing and perception',
    translations: { es: 'Audición y percepción' },
    collapsed: true,
    items: [
      { slug: 'perception', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Psychoacoustics',
        translations: { es: 'Psicoacústica' },
        collapsed: true,
        items: [
          { slug: 'perception/psychoacoustics', label: 'Overview', translations: { es: 'Resumen' } },
          'perception/psychoacoustics/loudness',
          'perception/psychoacoustics/advanced-loudness',
          'perception/psychoacoustics/sound-quality',
          'perception/psychoacoustics/tone-prominence',
          'perception/psychoacoustics/tone-audibility',
          'perception/psychoacoustics/psychoacoustic-annoyance',
        ],
      },
      {
        label: 'Speech',
        translations: { es: 'Habla' },
        collapsed: true,
        items: [
          { slug: 'perception/speech', label: 'Overview', translations: { es: 'Resumen' } },
          'perception/speech/speech-transmission',
          'perception/speech/speech-intelligibility',
          'perception/speech/objective-intelligibility',
        ],
      },
      {
        label: 'Hearing and exposure',
        translations: { es: 'Audición y exposición' },
        collapsed: true,
        items: [
          { slug: 'perception/hearing', label: 'Overview', translations: { es: 'Resumen' } },
          'perception/hearing/hearing-threshold',
          'perception/hearing/noise-induced-hearing-loss',
          'perception/hearing/occupational-exposure',
        ],
      },
    ],
  },
  {
    label: 'Rooms and buildings',
    translations: { es: 'Salas y edificación' },
    collapsed: true,
    items: [
      { slug: 'buildings', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Room acoustics',
        translations: { es: 'Acústica de salas' },
        collapsed: true,
        items: [
          { slug: 'buildings/rooms', label: 'Overview', translations: { es: 'Resumen' } },
          'buildings/rooms/room-impulse-response',
          'buildings/rooms/room-acoustics',
          'buildings/rooms/open-plan-acoustics',
          'buildings/rooms/room-image-sources',
          'buildings/rooms/room-noise',
          'buildings/rooms/reverberation-prediction',
          'buildings/rooms/enclosed-space-absorption',
        ],
      },
      {
        label: 'Sound insulation',
        translations: { es: 'Aislamiento acústico' },
        collapsed: true,
        items: [
          { slug: 'buildings/insulation', label: 'Overview', translations: { es: 'Resumen' } },
          'buildings/insulation/insulation-field',
          'buildings/insulation/insulation-lab',
          'buildings/insulation/insulation-intensity',
          'buildings/insulation/insulation-survey',
          'buildings/insulation/flanking-lab',
          'buildings/insulation/heavy-impact-sources',
          'buildings/insulation/insulation-ratings',
          'buildings/insulation/facade-insulation',
          'buildings/insulation/spanish-building-code',
        ],
      },
      {
        label: 'Insulation design',
        translations: { es: 'Diseño del aislamiento' },
        collapsed: true,
        items: [
          { slug: 'buildings/design', label: 'Overview', translations: { es: 'Resumen' } },
          'buildings/design/insulation-prediction',
          'buildings/design/detailed-prediction',
          'buildings/design/panel-sound-insulation',
          'buildings/design/impact-improvement',
          'buildings/design/structure-borne-power',
          'buildings/design/installed-structure-borne',
            ],
      },
    ],
  },
  {
    label: 'Materials and surfaces',
    translations: { es: 'Materiales y superficies' },
    collapsed: true,
    items: [
      { slug: 'materials', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Absorbers',
        translations: { es: 'Absorbentes' },
        collapsed: true,
        items: [
          { slug: 'materials/absorbers', label: 'Overview', translations: { es: 'Resumen' } },
          'materials/absorbers/absorption-measurement',
          'materials/absorbers/airflow-resistance',
          'materials/absorbers/impedance-tube',
          'materials/absorbers/porous-absorbers',
          'materials/absorbers/metamaterial-absorbers',
        ],
      },
      {
        label: 'Diffusers and surfaces',
        translations: { es: 'Difusores y superficies' },
        collapsed: true,
        items: [
          { slug: 'materials/diffusers', label: 'Overview', translations: { es: 'Resumen' } },
          'materials/diffusers/diffusers',
          'materials/diffusers/metadiffusers',
          'materials/surfaces/road-absorption',
        ],
      },
      {
        label: 'Resilient layers',
        translations: { es: 'Capas resilientes' },
        collapsed: true,
        items: [
          { slug: 'materials/resilient', label: 'Overview', translations: { es: 'Resumen' } },
          'materials/resilient/dynamic-stiffness',
        ],
      },
    ],
  },
  {
    label: 'Vibration and structure-borne sound',
    translations: { es: 'Vibración y ruido estructural' },
    collapsed: true,
    items: [
      { slug: 'vibration', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Structure-borne sources',
        translations: { es: 'Fuentes de ruido estructural' },
        collapsed: true,
        items: [
          { slug: 'vibration/structural', label: 'Overview', translations: { es: 'Resumen' } },
          'vibration/structural/mechanical-mobility',
          'vibration/structural/junction-transmission',
          'vibration/structural/transfer-stiffness',
          'devices/emission/vibration-sound-power',
                ],
      },
      {
        label: 'Human vibration',
        translations: { es: 'Vibración en humanos' },
        collapsed: true,
        items: [
          { slug: 'vibration/human', label: 'Overview', translations: { es: 'Resumen' } },
          'vibration/human/human-vibration',
          'vibration/human/multiple-shock-vibration',
        ],
      },
      {
        label: 'Machinery',
        translations: { es: 'Maquinaria' },
        collapsed: true,
        items: [
          { slug: 'vibration/machinery', label: 'Overview', translations: { es: 'Resumen' } },
          'vibration/machinery/machine-diagnostics',
        ],
      },
    ],
  },
  {
    label: 'Environment and transport',
    translations: { es: 'Medio ambiente y transporte' },
    collapsed: true,
    items: [
      { slug: 'environment', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Assessment and regulation',
        translations: { es: 'Evaluación y normativa' },
        collapsed: true,
        items: [
          { slug: 'environment/assessment', label: 'Overview', translations: { es: 'Resumen' } },
          'environment/environmental-levels',
          'environment/spanish-noise-regulation',
          'environment/assessment/impulsive-sound',
        ],
      },
      {
        label: 'Outdoor sound',
        translations: { es: 'Sonido en exteriores' },
        collapsed: true,
        items: [
          { slug: 'environment/propagation', label: 'Overview', translations: { es: 'Resumen' } },
          'environment/propagation/outdoor-propagation',
          'environment/sources/cnossos-rail-emission',
          'environment/sources/cnossos-road-emission',
          'environment/propagation/ground-barriers',
          'environment/propagation/atmospheric-refraction',
        ],
      },
      {
        label: 'Aircraft and wind energy',
        translations: { es: 'Aeronaves y energía eólica' },
        collapsed: true,
        items: [
          { slug: 'aircraft', label: 'Overview', translations: { es: 'Resumen' } },
          'aircraft/aircraft-noise',
          'aircraft/airport-noise',
          'aircraft/rotorcraft-noise',
          'environment/sources/wind-turbine-noise',
        ],
      },
    ],
  },
  {
    label: 'Underwater acoustics',
    translations: { es: 'Acústica submarina' },
    collapsed: true,
    items: [
      { slug: 'underwater', label: 'Overview', translations: { es: 'Resumen' } },
      'underwater/underwater-acoustics',
      'underwater/underwater-propagation',
      'underwater/underwater-solvers',
      'underwater/marine-mammal-exposure',
    ],
  },
  {
    label: 'Sources and devices',
    translations: { es: 'Fuentes y dispositivos' },
    collapsed: true,
    items: [
      { slug: 'devices', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Sound power and intensity',
        translations: { es: 'Potencia acústica e intensidad' },
        collapsed: true,
        items: [
          { slug: 'devices/emission', label: 'Overview', translations: { es: 'Resumen' } },
          'devices/emission/intensity',
          'devices/emission/sound-power',
          'devices/emission/sound-power-pressure',
          'devices/emission/sound-power-reverberation',
          'devices/emission/sound-power-intensity',
        ],
      },
      {
        label: 'Electroacoustics',
        translations: { es: 'Electroacústica' },
        collapsed: true,
        items: [
          { slug: 'devices/electroacoustics', label: 'Overview', translations: { es: 'Resumen' } },
          'devices/electroacoustics/electroacoustics',
          'devices/electroacoustics/loudspeakers',
          'devices/electroacoustics/microphones',
          'devices/electroacoustics/swept-sine-distortion',
          'devices/broadcast/program-loudness',
        ],
      },
      {
        label: 'Noise control',
        translations: { es: 'Control de ruido' },
        collapsed: true,
        items: [
          { slug: 'devices/noise-control', label: 'Overview', translations: { es: 'Resumen' } },
          'devices/noise-control/silencers',
          'devices/noise-control/duct-path',
          'devices/noise-control/room-to-room',
          'devices/noise-control/noise-control',
        ],
      },
    ],
  },
  {
    label: 'Wave simulation',
    translations: { es: 'Simulación de ondas' },
    collapsed: true,
    items: [
      { slug: 'simulation', label: 'Overview', translations: { es: 'Resumen' } },
      'simulation/fdtd-simulation',
      'simulation/elastic-waves',
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
