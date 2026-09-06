/**
 * The site's navigation, one sidebar per topic.
 *
 * Every page belongs to a topic, the topic owns the sidebar a reader sees, and
 * `starlight-sidebar-topics` renders the topic list above it. That is what
 * makes the tree readable: an underwater page shows four guides and its own
 * API branch instead of every page in the library folded three deep.
 *
 * The API reference of a domain lives inside that domain's topic, which the
 * taxonomy makes possible: a section of the generated reference is a
 * subpackage is a topic. `apiSections` comes from
 * `scripts/generate_api_docs.py`, one group per section, so adding a module
 * changes nothing here. The last topic is a link to the global index, for a
 * reader who wants the whole table rather than one domain.
 *
 * `astro.config.mjs` hands this array to the plugin and
 * `scripts/check-sidebar.mjs` reads it to work out what the rendered tree has
 * to look like, so the checks follow the content.
 */
import { apiSections } from '../generated/api-sidebar.mjs';

const apiGroup = (...keys) => ({
  label: 'API reference',
  translations: { es: 'Referencia de la API' },
  collapsed: true,
  items: [
    // The landing row of the group, as every other group has one: it is what
    // the breadcrumb trail of a module page hangs from, and the way back to
    // the whole table from inside one domain.
    { slug: 'reference/api', label: 'Overview', translations: { es: 'Resumen' } },
    // A key that names no section means this file and the generator have
    // drifted apart. Without this the group takes an `undefined` where a
    // subtree should be, which Starlight renders as nothing at all.
    ...keys.map((key) => {
      const section = apiSections[key];
      if (!section) throw new Error(`apiGroup: no API section named "${key}"`);
      return section;
    }),
  ],
});

export const topics = [
  {
    id: 'start',
    label: { en: 'Start', es: 'Inicio' },
    link: '/start/',
    items: [
      { slug: 'start', label: 'Overview', translations: { es: 'Resumen' } },
      'start/getting-started',
      // The task-shaped door. The rest of the site is a subject tree, which
      // only helps a reader who already knows which subject owns their job;
      // this row is the same library indexed by the job instead.
      { slug: 'start/tasks', label: 'What do you need to measure?', translations: { es: '¿Qué necesitas medir?' } },
      // The map of the eleven topics. A sibling entry point rather than a parent:
      // the area overviews stay the breadcrumb ancestors of their own guides.
      { slug: 'start/guides', label: 'All guides', translations: { es: 'Todas las guías' } },
      'start/why-phonometry',
      'start/about',
      'start/support',
    ],
  },
  {
    id: 'signals',
    label: { en: 'Signal analysis', es: 'Análisis de señal' },
    link: '/signals/',
    items: [
      { slug: 'signals', label: 'Overview', translations: { es: 'Resumen' } },
      'signals/sound-level-meter',
      {
        label: 'Octave filtering',
        translations: { es: 'Filtrado en octavas' },
        collapsed: true,
        items: [
          { slug: 'signals/filters', label: 'Overview', translations: { es: 'Resumen' } },
          'signals/filters/filter-banks',
          'signals/filters/filter-gallery',
          'signals/filters/filter-compliance',
          'signals/filters/block-processing',
          'signals/filters/multichannel',
        ],
      },
      {
        label: 'Levels and weighting',
        translations: { es: 'Niveles y ponderación' },
        collapsed: true,
        items: [
          { slug: 'signals/levels', label: 'Overview', translations: { es: 'Resumen' } },
          'signals/levels/weighting',
          'signals/levels/special-weightings',
          'signals/levels/time-weighting',
          'signals/levels/levels',
        ],
      },
      {
        label: 'Signals and spectra',
        translations: { es: 'Señales y espectros' },
        collapsed: true,
        items: [
          { slug: 'signals/spectra', label: 'Overview', translations: { es: 'Resumen' } },
          'signals/spectra/spectral-analysis',
          'signals/spectra/miso-coherence',
          'signals/spectra/time-frequency',
          'signals/spectra/cepstrum-echoes',
          'signals/spectra/synchronous-averaging',
          'signals/spectra/correlation-delay',
          'signals/spectra/test-signals',
          'signals/spectra/system-measurement',
        ],
      },
      {
        label: 'Calibration and uncertainty',
        translations: { es: 'Calibración e incertidumbre' },
        collapsed: true,
        items: [
          { slug: 'signals/metrology', label: 'Overview', translations: { es: 'Resumen' } },
          'signals/metrology/calibration',
          'signals/metrology/compliance-verification',
          'signals/metrology/gum-uncertainty',
          'signals/metrology/data-qualification',
        ],
      },
      apiGroup('filters', 'signals', 'metrology'),
    ],
  },
  {
    id: 'fluids',
    label: { en: 'The medium', es: 'El medio' },
    link: '/fluids/',
    items: [
      { slug: 'fluids', label: 'Overview', translations: { es: 'Resumen' } },
      'fluids/humid-air',
      apiGroup('fluids'),
    ],
  },
  {
    id: 'io',
    label: { en: 'Audio files', es: 'Archivos de audio' },
    link: '/io/',
    items: [
      { slug: 'io', label: 'Overview', translations: { es: 'Resumen' } },
      'io/audio-files',
      apiGroup('io'),
    ],
  },
  {
    id: 'perception',
    label: { en: 'Hearing and perception', es: 'Audición y percepción' },
    link: '/perception/',
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
          'perception/hearing/hearing-protectors',
        ],
      },
      apiGroup('psychoacoustics', 'speech', 'hearing'),
    ],
  },
  {
    id: 'buildings',
    label: { en: 'Rooms and buildings', es: 'Salas y edificación' },
    link: '/buildings/',
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
          'buildings/rooms/sound-strength',
          'buildings/rooms/spatial-impression',
          'buildings/rooms/stage-and-uncertainty',
          'buildings/rooms/reporting-and-qualification',
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
          'buildings/insulation/low-frequency-procedure',
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
      'buildings/design/resilient-layers',
          'buildings/design/structure-borne-power',
          'buildings/design/installed-structure-borne',
            ],
      },
      apiGroup('rooms', 'building'),
    ],
  },
  {
    id: 'materials',
    label: { en: 'Materials and surfaces', es: 'Materiales y superficies' },
    link: '/materials/',
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
          { slug: 'materials/surfaces', label: 'Surfaces overview', translations: { es: 'Resumen de superficies' } },
          'materials/surfaces/road-absorption',
        ],
      },
      {
        label: 'Resilient layers',
        translations: { es: 'Capas elásticas' },
        collapsed: true,
        items: [
          { slug: 'materials/resilient', label: 'Overview', translations: { es: 'Resumen' } },
          'materials/resilient/dynamic-stiffness',
        ],
      },
      apiGroup('materials'),
    ],
  },
  {
    id: 'vibration',
    label: { en: 'Vibration and structure-borne sound', es: 'Vibración y ruido estructural' },
    link: '/vibration/',
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
          'vibration/machinery/machine-vibration-evaluation',
        ],
      },
      apiGroup('vibration'),
    ],
  },
  {
    id: 'environment',
    label: { en: 'Environment and transport', es: 'Medio ambiente y transporte' },
    link: '/environment/',
    items: [
      { slug: 'environment', label: 'Overview', translations: { es: 'Resumen' } },
      {
        label: 'Assessment and regulation',
        translations: { es: 'Evaluación y normativa' },
        collapsed: true,
        items: [
          { slug: 'environment/assessment', label: 'Overview', translations: { es: 'Resumen' } },
          'environment/assessment/environmental-levels',
          'environment/assessment/spanish-noise-regulation',
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
          'environment/propagation/ground-barriers',
          'environment/propagation/atmospheric-refraction',
        ],
      },
      {
        label: 'Sources',
        translations: { es: 'Fuentes' },
        collapsed: true,
        items: [
          { slug: 'environment/sources', label: 'Overview', translations: { es: 'Resumen' } },
          'environment/sources/cnossos-road-emission',
          'environment/sources/cnossos-rail-emission',
          'environment/sources/wind-turbine-noise',
        ],
      },
      apiGroup('environment'),
    ],
  },
  {
    id: 'aircraft',
    label: { en: 'Aircraft noise', es: 'Ruido de aeronaves' },
    link: '/aircraft/',
    items: [
      { slug: 'aircraft', label: 'Overview', translations: { es: 'Resumen' } },
      'aircraft/aircraft-noise',
      'aircraft/airport-noise',
      'aircraft/anp-fleet',
      'aircraft/rotorcraft-noise',
      apiGroup('aeroacoustics'),
    ],
  },
  {
    id: 'underwater',
    label: { en: 'Underwater acoustics', es: 'Acústica submarina' },
    link: '/underwater/',
    items: [
      { slug: 'underwater', label: 'Overview', translations: { es: 'Resumen' } },
      'underwater/underwater-acoustics',
      'underwater/underwater-propagation',
      'underwater/underwater-solvers',
      'underwater/marine-mammal-exposure',
      apiGroup('underwater'),
    ],
  },
  {
    id: 'devices',
    label: { en: 'Sources and devices', es: 'Fuentes y dispositivos' },
    link: '/devices/',
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
          'devices/emission/sound-power-in-situ',
          'devices/emission/sound-power-in-duct',
          'devices/emission/vibration-sound-power',
          'devices/emission/sound-power-intensity',
          'devices/emission/workstation-emission',
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
          { slug: 'devices/broadcast', label: 'Broadcast overview', translations: { es: 'Resumen de radiodifusión' } },
          'devices/broadcast/program-loudness',
          'devices/broadcast/quasi-peak',
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
          'devices/noise-control/vdi2081-air-systems',
          'devices/noise-control/room-to-room',
          'devices/noise-control/noise-control',
        ],
      },
      apiGroup('power', 'electroacoustics', 'broadcast', 'noise_control'),
    ],
  },
  {
    id: 'simulation',
    label: { en: 'Wave simulation', es: 'Simulación de ondas' },
    link: '/simulation/',
    items: [
      { slug: 'simulation', label: 'Overview', translations: { es: 'Resumen' } },
      'simulation/fdtd-simulation',
      'simulation/elastic-waves',
      apiGroup('simulation'),
    ],
  },
  {
    id: 'reference',
    label: { en: 'Reference', es: 'Referencia' },
    link: '/reference/',
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
      // The third leg of the evidence story: conformance says what the
      // numbers are checked against, errata where the print is wrong, and
      // this one why the numbers read the same on every machine.
      {
        slug: 'reference/determinism',
        label: 'Same bits on every machine',
        translations: { es: 'Los mismos bits en cualquier máquina' },
      },
      'reference/bibliography',
      'reference/glossary',
    ],
  },
  {
    id: 'api',
    label: { en: 'API reference', es: 'Referencia de la API' },
    link: '/reference/api/',
    // Every section, in the generated order. This is the destination of the
    // "Overview" row inside each domain's API group, so it has to be worth the
    // sidebar it swaps in: the whole table, not a single row pointing at the
    // page the reader is already on.
    items: [
      { slug: 'reference/api', label: 'The whole API', translations: { es: 'Toda la API' } },
      ...Object.values(apiSections),
    ],
  },
];

/**
 * The topics as a plain group tree, for everything that reads the navigation
 * rather than renders it: the breadcrumb trails, the per-area OG artwork and
 * the sidebar behaviour check. A topic is a top-level group, which is what
 * those three already assumed when the whole site shared one sidebar.
 */
export const navigationTree = topics
  .filter((topic) => topic.items)
  .map((topic) => ({
    label: topic.label.en,
    translations: { es: topic.label.es },
    collapsed: true,
    items: topic.items,
  }));

/**
 * The three topics that are not a domain of acoustics: the entry point, the
 * reference shelf and the generated API tree. Everything else teaches, which
 * is what makes its pages guides.
 */
const NOT_A_DOMAIN = new Set(['start', 'reference', 'api']);

/**
 * The topics as URL prefixes, for everything that has to answer "which topic
 * owns this page?" from a slug alone: the `articleSection` of the structured
 * data and the article type that says the page teaches.
 *
 * The prefix comes from the topic's own `link`, not from its id, because the
 * two differ where a topic lives inside another one's path: the API tree is
 * the topic `api` and it lives at `reference/api`. Sorted longest first, so a
 * page under `reference/api/` resolves to the API topic rather than to
 * Reference, whose prefix it also starts with.
 *
 * @type {{id: string, prefix: string, label: {en: string, es: string}, isDomain: boolean}[]}
 */
export const topicSections = topics
  .map((topic) => ({
    id: topic.id,
    prefix: topic.link.replace(/^\/+|\/+$/g, ''),
    label: topic.label,
    isDomain: !NOT_A_DOMAIN.has(topic.id),
  }))
  .sort((a, b) => b.prefix.length - a.prefix.length);

/**
 * The ids of the topics that teach, in tree order. They are also the content
 * directory names, since a domain topic's link is its id.
 */
export const domainIds = topics
  .filter((topic) => !NOT_A_DOMAIN.has(topic.id))
  .map((topic) => topic.id);

/**
 * The topic that owns a slug, or `undefined` for a page outside every topic
 * (the home page and the 404).
 *
 * @param {string} slug Page slug without the locale prefix, e.g. `signals/levels/weighting`.
 */
export function topicForSlug(slug) {
  return topicSections.find(
    (topic) => slug === topic.prefix || slug.startsWith(`${topic.prefix}/`),
  );
}
