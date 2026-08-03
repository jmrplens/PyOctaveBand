/**
 * Landing-page content, one object per locale.
 *
 * The two locales share this single shape, so a block cannot exist in one
 * language and quietly go missing in the other; index.mdx and es/index.mdx
 * import from here and pass the object down to the components in
 * src/components/home/.
 *
 * Every number in `stats` and every claim in `honest` is taken from the
 * repository, not written by hand from memory:
 *   checks/domains/standards   conformance-stats.mjs, parsed from
 *                              docs/CONFORMANCE.md at build time
 *   guides/apiPages/figures/fiches/version   repo-stats.mjs, counted from the
 *                              tree at build time (definitions in that module)
 *
 * All of these used to be typed out here by hand, and every hand-written copy
 * fell a release behind. They are imported now, so there is exactly one place
 * that can be wrong.
 */
import { checksRatio, domains, standards } from './conformance-stats.mjs';
import { apiPages, fiches, figures, guides, version } from './repo-stats.mjs';

export interface Stat {
	value: string;
	label: string;
	href?: string;
}

export interface Area {
	name: string;
	href: string;
	standards: string[];
	summary: string;
}

export interface Step {
	title: string;
	body: string;
	code?: string;
	href: string;
	linkText: string;
}

export interface HomeContent {
	/** Accessible name for the numbers row, which has no visible heading. */
	statsLabel: string;
	stats: Stat[];
	what: { title: string; body: string[] };
	who: { title: string; items: string[] };
	honest: { title: string; items: string[] };
	proof: {
		title: string;
		lead: string;
		spectrum: { title: string; note: string; code: string; alt: string; caption: string };
		fiche: { title: string; note: string; code: string; caption: string; description: string; linkTitle: string };
	};
	coverage: { title: string; lead: string; areas: Area[]; headers: [string, string] };
	start: { title: string; lead: string; steps: Step[] };
}

/**
 * The concept DOI of the Zenodo archive, the one that always resolves to the
 * latest release. Stated once here; `astro.config.mjs` and the About page
 * carry the same value in their own machine-readable and citation contexts.
 */
const DOI = '10.5281/zenodo.21215280';

const SPECTRUM_CODE = `import numpy as np
from phonometry import filters

fs = 48_000
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1000 * t)

spl, freq = filters.octave_filter(signal, fs=fs, fraction=3)`;

// The call that renders the committed fiche shown beside it
// (scripts/generate_reports.py), with the metadata block cut to the four
// fields the page discusses. Same inputs, same Rw (C; Ctr) = 31 (-1; -3) dB
// and the same PASS against the 30 dB requirement.
const FICHE_CODE = `import numpy as np
from phonometry import building, ReportMetadata

freqs = np.array([100, 125, 160, 200, 250, 315,
                  400, 500, 630, 800, 1000, 1250,
                  1600, 2000, 2500, 3150])

R = building.single_panel_transmission_loss(
    freqs, mass_per_area=15.0,
    critical_frequency=2000.0, loss_factor=0.02,
)

meta = ReportMetadata(
    specimen="6 mm float glass pane",
    laboratory="Phonometry reference example",
    report_id="EXAMPLE-717-1",
    requirement=30.0,  # adds the verdict row
)

R.report("Rw_fiche.pdf", metadata=meta)`;

export const en: HomeContent = {
	statsLabel: 'The library in four numbers',
	stats: [
		{ value: checksRatio, label: 'conformance checks passing', href: '/phonometry/reference/conformance/' },
		{ value: String(standards), label: `standards referenced, across ${domains} domains` },
		{ value: String(figures), label: 'figures, each in light and dark, English and Spanish' },
		{ value: String(fiches), label: 'PDF fiches in the reporting format their standard defines, rendered by .report()' },
	],
	what: {
		title: 'What this is',
		body: [
			'phonometry is a Python library for acoustic measurement, analysis and prediction. You give it a signal, a measured spectrum or a set of geometrical and material inputs; it gives you the quantity the standard defines, with the intermediate terms still visible.',
			'Every result is a typed, frozen dataclass. It carries the inputs it was computed from, it draws its own figure with a one-line <code>.plot()</code> in English or Spanish, and where a standard defines a reporting format it renders a one-page PDF fiche with <code>.report()</code>.',
			`It is written and maintained by one person, published under the MIT licence on <a href="https://pypi.org/project/phonometry/">PyPI</a>, archived with a <a href="https://doi.org/${DOI}">DOI on Zenodo</a>, and currently at version ${version}. It needs Python 3.13 or newer with NumPy and SciPy; matplotlib, numba and reportlab are optional extras.`,
		],
	},
	who: {
		title: 'Who it is for',
		items: [
			'Consultants and test laboratories who need the number a standard defines and the terms behind it, in a form that can go into a report.',
			'Researchers who need a readable, citable reference implementation rather than a black box, and who need to see which clause each formula came from.',
			'Audio, firmware and product engineers who need class 1 weighting and filters, distortion, loudness and speech metrics inside their own test rigs.',
			'Students and teachers, since every guide states the standard, the formula and the assumptions before the code.',
		],
	},
	honest: {
		title: 'What it is not',
		items: [
			'Not a certified instrument. The conformance report is the library checking its own output against values published in the standards, not an accredited third-party calibration. Legal measurements still need type-approved hardware.',
			'Not a data-acquisition system. It processes arrays and files you supply; it does not talk to sound cards, microphones or analysers.',
			'Not a complete implementation of every standard it cites. Each guide states which clauses, methods and annexes are covered, and which are not.',
		],
	},
	proof: {
		title: 'What it looks like in use',
		lead: 'Two things the library does that are hard to claim and easy to show: the result of an analysis draws itself, and where a standard prescribes a reporting layout, the same result renders that layout as a PDF.',
		spectrum: {
			title: 'An analysis, and the figure it draws',
			note: 'One-third-octave band levels of a two-tone signal, per IEC 61260-1 band edges, with the raw power spectral density behind them. The figure below is the one committed to the documentation, not a mock-up.',
			code: SPECTRUM_CODE,
			alt: 'One-third-octave spectrum of a two-tone signal, with the raw power spectral density in the background',
			caption: 'From the Getting Started guide: 33 bands from 12.6 Hz to 20 kHz, peaking at 100 Hz and 1 kHz.',
		},
		fiche: {
			title: 'A rating, and the fiche it renders',
			note: 'The weighted airborne rating of ISO 717-1 over the 16 one-third-octave bands the rating uses, rendered in a laboratory report layout: metadata header, band table, the curve against the shifted reference, the boxed single-number result and the verdict against the requirement. The code is the call that renders the fiche beside it, with the metadata block cut to the fields discussed here.',
			code: FICHE_CODE,
			caption: 'Airborne rating fiche (SoundReductionResult.report), Rw (C; Ctr).',
			linkTitle: 'Airborne ISO 717-1 example report (PDF)',
			description:
				'One-page airborne sound insulation fiche for a 6 mm float glass pane: a metadata header, the one-third-octave R table beside the plot of that curve against the shifted reference, the boxed result Rw (C; Ctr) = 31 (-1; -3) dB and a PASS verdict against the 30 dB requirement.',
		},
	},
	coverage: {
		title: 'What it covers',
		lead: `Nine areas, ${guides} guides, ${apiPages} API reference pages. Each row links to the area overview; the designations are the standards actually implemented there, not a reading list.`,
		headers: ['Area', 'Standards implemented'],
		areas: [
			{
				name: 'Core signal analysis',
				href: '/phonometry/signal/',
				summary: 'Filter banks, weighting, levels, spectra, calibration and uncertainty.',
				standards: ['IEC 61260-1', 'ANSI S1.11', 'IEC 61672-1', 'ISO 7196', 'IEC 61252', 'ISO 1996-1', 'IEC 60942', 'GUM'],
			},
			{
				name: 'Hearing and perception',
				href: '/phonometry/perception/',
				summary: 'Loudness, sound quality, speech intelligibility, hearing and exposure.',
				standards: ['ISO 532-1/-2/-3', 'ECMA-418-1/-2', 'ISO 226', 'DIN 45692', 'IEC 60268-16', 'ANSI S3.5', 'DIN 45681', 'ISO/PAS 20065', 'ISO 7029', 'ISO 389-7', 'ISO 1999', 'ISO 9612'],
			},
			{
				name: 'Rooms and buildings',
				href: '/phonometry/buildings/',
				summary: 'Room parameters, background noise, field and laboratory insulation, prediction.',
				standards: ['ISO 3382-1/-2/-3', 'ISO 16283-1/-2/-3', 'ISO 10140', 'ISO 10848', 'ISO 15186-1/-2', 'ISO 16251-1', 'ISO 717-1/-2', 'EN 12354-1…-6', 'ISO 18233', 'ISO 12999-1', 'ISO 10052', 'ANSI/ASA S12.2', 'ASTM E413/E1414'],
			},
			{
				name: 'Materials and surfaces',
				href: '/phonometry/materials/',
				summary: 'Absorption, airflow resistance, impedance tube, porous and metamaterial models, diffusers, scattering.',
				standards: ['ISO 354', 'ISO 11654', 'ISO 10534-1/-2', 'ISO 9053-1/-2', 'ISO 17497-1/-2', 'ISO 13472-1/-2', 'EN 29052-1', 'ISO 12999-2'],
			},
			{
				name: 'Vibration and structure-borne sound',
				href: '/phonometry/vibration/',
				summary: 'Mobility and FRFs, isolators, radiated power, junctions, human vibration.',
				standards: ['ISO 7626-1/-2', 'ISO 10846-1/-2/-3', 'ISO 9611', 'ISO/TS 7849-1/-2', 'EN 15657', 'EN 12354-5', 'ISO 2631-1/-2/-4/-5', 'ISO 5349-1/-2', 'ISO 8041-1'],
			},
			{
				name: 'Environment and transport',
				href: '/phonometry/environment/',
				summary: 'Outdoor propagation, barriers, refraction, aircraft, rotorcraft and wind turbines.',
				standards: ['ISO 9613-1/-2', 'ISO 1996-1/-2', 'ISO/PAS 1996-3', 'NT ACOU 112', 'CNOSSOS-EU (2002/49/EC Annex II)', 'ICAO Annex 16', 'IEC 61265', 'SAE ARP 5534', 'ECAC Doc 29/32', 'IEC 61400-11'],
			},
			{
				name: 'Underwater acoustics',
				href: '/phonometry/underwater/',
				summary: 'Levels re 1 µPa, ship radiated noise, pile driving, ambient noise, transmission loss.',
				standards: ['ISO 18405', 'ISO 17208-1/-2', 'ISO 18406', 'JOMOPANS-ECHO'],
			},
			{
				name: 'Sources and devices',
				href: '/phonometry/devices/',
				summary: 'Sound power, intensity, emission declarations, electroacoustics, programme loudness.',
				standards: ['ISO 3741', 'ISO 3744/3746', 'ISO 3745', 'ISO 9614-1/-2/-3', 'IEC 61043', 'ISO 4871', 'IEC 60268-3/-4/-5', 'ITU-R BS.1770-5', 'EBU R 128'],
			},
			{
				name: 'Wave simulation',
				href: '/phonometry/simulation/',
				summary: 'Deterministic 2D FDTD solvers, acoustic and elastic P-SV, validated against analytic oracles rather than a standard.',
				standards: ['no governing standard'],
			},
		],
	},
	start: {
		title: 'Starting from zero',
		lead: 'Three steps, in order. The first two take a few minutes; the third is the part you keep coming back to.',
		steps: [
			{
				title: 'Install it',
				body: 'Python 3.13 or newer. The base install pulls in NumPy and SciPy; the full extra adds plotting, the faster impulse ballistics and PDF fiche rendering.',
				code: 'pip install phonometry[full]',
				href: '/phonometry/start/getting-started/',
				linkText: 'Installation options',
			},
			{
				title: 'Run one analysis end to end',
				body: 'Getting Started walks the whole processing chain once, on a synthetic signal and then on a WAV file: calibration, frequency weighting, the filter bank, time weighting and the levels that come out.',
				href: '/phonometry/start/getting-started/',
				linkText: 'Getting Started',
			},
			{
				title: 'Go to your own domain',
				body: 'Each guide opens with the standard it implements, the quantities it defines and the assumptions, then the code, then the figure. When you need the exact signature, the API reference has one page per module.',
				href: '/phonometry/reference/api/',
				linkText: 'API reference',
			},
		],
	},
};

export const es: HomeContent = {
	statsLabel: 'La biblioteca en cuatro cifras',
	stats: [
		{ value: checksRatio, label: 'comprobaciones de conformidad superadas', href: '/phonometry/es/reference/conformance/' },
		{ value: String(standards), label: `normas referenciadas, en ${domains} dominios` },
		{ value: String(figures), label: 'figuras, cada una en claro y oscuro, en inglés y español' },
		{ value: String(fiches), label: 'fichas PDF con el formato de informe que define su norma, generadas por .report()' },
	],
	what: {
		title: 'Qué es',
		body: [
			'phonometry es una biblioteca de Python para medición, análisis y predicción acústica. Le das una señal, un espectro medido o un conjunto de datos geométricos y de materiales, y te devuelve la magnitud que define la norma, con los términos intermedios a la vista.',
			'Cada resultado es un dataclass tipado y congelado (<code>frozen</code>). Conserva los datos de entrada con los que se calculó, dibuja su propia figura con un <code>.plot()</code> de una línea en inglés o en español y, cuando una norma define un formato de informe, genera una ficha PDF de una página con <code>.report()</code>.',
			`La escribe y la mantiene una sola persona, se publica con licencia MIT en <a href="https://pypi.org/project/phonometry/">PyPI</a>, se archiva con <a href="https://doi.org/${DOI}">DOI en Zenodo</a> y va por la versión ${version}. Necesita Python 3.13 o posterior con NumPy y SciPy; matplotlib, numba y reportlab son extras opcionales.`,
		],
	},
	who: {
		title: 'Para quién es',
		items: [
			'Consultores y laboratorios de ensayo que necesitan el número que define la norma y los términos que hay detrás, en una forma que se pueda llevar a un informe.',
			'Investigadores que necesitan una implementación de referencia legible y citable en lugar de una caja negra, y ver de qué apartado sale cada fórmula.',
			'Ingenieros de audio, firmware y producto que necesitan filtros y ponderaciones de clase 1, distorsión, sonoridad y métricas de habla dentro de sus propios bancos de ensayo.',
			'Estudiantes y docentes, porque cada guía enuncia la norma, la fórmula y las hipótesis antes del código.',
		],
	},
	honest: {
		title: 'Qué no es',
		items: [
			'No es un instrumento certificado. El informe de conformidad consiste en que la biblioteca comprueba su propia salida frente a valores publicados en las normas; no es una calibración acreditada por un tercero. Una medición con validez legal sigue necesitando equipo con aprobación de modelo.',
			'No es un sistema de adquisición de datos. Procesa los arrays y los archivos que le pases; no habla con tarjetas de sonido, micrófonos ni analizadores.',
			'No implementa por completo todas las normas que cita. Cada guía indica qué apartados, métodos y anexos cubre y cuáles no.',
		],
	},
	proof: {
		title: 'Cómo se ve en uso',
		lead: 'Dos cosas que hace la biblioteca, difíciles de afirmar y fáciles de enseñar: el resultado de un análisis se dibuja solo y, cuando una norma prescribe un formato de informe, ese mismo resultado lo genera en PDF.',
		spectrum: {
			title: 'Un análisis y la figura que dibuja',
			note: 'Niveles por bandas de tercio de octava de una señal de dos tonos, con los límites de banda de IEC 61260-1 y la densidad espectral de potencia bruta al fondo. La figura de abajo es la que está publicada en la documentación, no una maqueta.',
			code: SPECTRUM_CODE,
			alt: 'Espectro en tercios de octava de una señal de dos tonos, con la densidad espectral de potencia bruta al fondo',
			caption: 'De la guía Primeros pasos: 33 bandas de 12,6 Hz a 20 kHz, con máximos en 100 Hz y 1 kHz.',
		},
		fiche: {
			title: 'Un índice global y la ficha que genera',
			note: 'El índice global de aislamiento a ruido aéreo de ISO 717-1 sobre las 16 bandas de tercio de octava que usa el índice, con el formato de un informe de laboratorio: cabecera de metadatos, tabla por bandas, la curva frente a la curva de referencia desplazada, el resultado de un solo número enmarcado y el veredicto frente al requisito. El código es la llamada que genera la ficha de al lado, con el bloque de metadatos reducido a los campos de los que se habla aquí.',
			code: FICHE_CODE,
			caption: 'Ficha del índice global a ruido aéreo (SoundReductionResult.report), Rw (C; Ctr).',
			linkTitle: 'Informe de ejemplo ISO 717-1 a ruido aéreo (PDF)',
			description:
				'Ficha de aislamiento a ruido aéreo de una página para un vidrio flotado de 6 mm: cabecera de metadatos, tabla de R en tercios de octava junto al gráfico de esa curva frente a la de referencia desplazada, el resultado Rw (C; Ctr) = 31 (-1; -3) dB enmarcado y el veredicto (PASS, en inglés como el resto de la ficha) frente al requisito de 30 dB.',
		},
	},
	coverage: {
		title: 'Qué abarca',
		lead: `Nueve áreas, ${guides} guías y ${apiPages} páginas de referencia de la API. Cada fila enlaza al índice del área; las designaciones son las normas realmente implementadas ahí, no una lista de lecturas.`,
		headers: ['Área', 'Normas implementadas'],
		areas: [
			{
				name: 'Análisis de señal',
				href: '/phonometry/es/signal/',
				summary: 'Bancos de filtros, ponderaciones, niveles, espectros, calibración e incertidumbre.',
				standards: ['IEC 61260-1', 'ANSI S1.11', 'IEC 61672-1', 'ISO 7196', 'IEC 61252', 'ISO 1996-1', 'IEC 60942', 'GUM'],
			},
			{
				name: 'Audición y percepción',
				href: '/phonometry/es/perception/',
				summary: 'Sonoridad, calidad sonora, inteligibilidad del habla, audición y exposición.',
				standards: ['ISO 532-1/-2/-3', 'ECMA-418-1/-2', 'ISO 226', 'DIN 45692', 'IEC 60268-16', 'ANSI S3.5', 'DIN 45681', 'ISO/PAS 20065', 'ISO 7029', 'ISO 389-7', 'ISO 1999', 'ISO 9612'],
			},
			{
				name: 'Salas y edificación',
				href: '/phonometry/es/buildings/',
				summary: 'Parámetros de sala, ruido de fondo, aislamiento en campo y laboratorio, predicción.',
				standards: ['ISO 3382-1/-2/-3', 'ISO 16283-1/-2/-3', 'ISO 10140', 'ISO 10848', 'ISO 15186-1/-2', 'ISO 16251-1', 'ISO 717-1/-2', 'EN 12354-1…-6', 'ISO 18233', 'ISO 12999-1', 'ISO 10052', 'ANSI/ASA S12.2', 'ASTM E413/E1414'],
			},
			{
				name: 'Materiales y superficies',
				href: '/phonometry/es/materials/',
				summary: 'Absorción, resistencia al flujo de aire, tubo de impedancia, modelos porosos y de metamaterial, difusores, dispersión.',
				standards: ['ISO 354', 'ISO 11654', 'ISO 10534-1/-2', 'ISO 9053-1/-2', 'ISO 17497-1/-2', 'ISO 13472-1/-2', 'EN 29052-1', 'ISO 12999-2'],
			},
			{
				name: 'Vibración y ruido estructural',
				href: '/phonometry/es/vibration/',
				summary: 'Movilidad y FRF, aisladores, potencia radiada, uniones, vibración en humanos.',
				standards: ['ISO 7626-1/-2', 'ISO 10846-1/-2/-3', 'ISO 9611', 'ISO/TS 7849-1/-2', 'EN 15657', 'EN 12354-5', 'ISO 2631-1/-2/-4/-5', 'ISO 5349-1/-2', 'ISO 8041-1'],
			},
			{
				name: 'Medio ambiente y transporte',
				href: '/phonometry/es/environment/',
				summary: 'Propagación en exteriores, barreras, refracción, aeronaves, helicópteros y aerogeneradores.',
				standards: ['ISO 9613-1/-2', 'ISO 1996-1/-2', 'ISO/PAS 1996-3', 'NT ACOU 112', 'CNOSSOS-EU (Directiva 2002/49/CE, anexo II)', 'Anexo 16 OACI', 'IEC 61265', 'SAE ARP 5534', 'ECAC Doc 29/32', 'IEC 61400-11'],
			},
			{
				name: 'Acústica submarina',
				href: '/phonometry/es/underwater/',
				summary: 'Niveles re 1 µPa, ruido radiado por buques, hincado de pilotes, ruido ambiente, pérdidas por transmisión.',
				standards: ['ISO 18405', 'ISO 17208-1/-2', 'ISO 18406', 'JOMOPANS-ECHO'],
			},
			{
				name: 'Fuentes y dispositivos',
				href: '/phonometry/es/devices/',
				summary: 'Potencia acústica, intensidad, declaraciones de emisión, electroacústica, sonoridad de programa.',
				standards: ['ISO 3741', 'ISO 3744/3746', 'ISO 3745', 'ISO 9614-1/-2/-3', 'IEC 61043', 'ISO 4871', 'IEC 60268-3/-4/-5', 'ITU-R BS.1770-5', 'EBU R 128'],
			},
			{
				name: 'Simulación de ondas',
				href: '/phonometry/es/simulation/',
				summary: 'Solvers FDTD 2D deterministas, acústicos y elásticos P-SV, validados frente a oráculos analíticos y no frente a una norma.',
				standards: ['sin norma que la rija'],
			},
		],
	},
	start: {
		title: 'Empezar desde cero',
		lead: 'Tres pasos, en orden. Los dos primeros son cuestión de minutos; al tercero vuelves una y otra vez.',
		steps: [
			{
				title: 'Instálala',
				body: 'Python 3.13 o posterior. La instalación básica trae NumPy y SciPy; el extra completo añade las figuras, la ponderación temporal de impulso acelerada y la generación de fichas PDF.',
				code: 'pip install phonometry[full]',
				href: '/phonometry/es/start/getting-started/',
				linkText: 'Opciones de instalación',
			},
			{
				title: 'Haz un análisis de principio a fin',
				body: 'La guía Primeros pasos recorre una vez toda la cadena de proceso, primero con una señal sintética y después con un archivo WAV: calibración, ponderación frecuencial, banco de filtros, ponderación temporal y los niveles que salen.',
				href: '/phonometry/es/start/getting-started/',
				linkText: 'Primeros pasos',
			},
			{
				title: 'Ve a tu dominio',
				body: 'Cada guía empieza por la norma que implementa, las magnitudes que define y las hipótesis, y sigue con el código y la figura. Cuando necesites la firma exacta, la referencia de la API tiene una página por módulo.',
				href: '/phonometry/es/reference/api/',
				linkText: 'Referencia de la API',
			},
		],
	},
};

export const home = { en, es };
