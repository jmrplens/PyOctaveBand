/**
 * Derivation of the page header chips from the typed `references` frontmatter
 * block (schema in src/content.config.ts), shared by PageChips.astro (which
 * renders the run under the H1) and References.astro (which stamps the
 * matching anchors on the bibliography entries).
 *
 * The frontmatter bibliography is the single source of truth: the header and
 * the References section are two views of the same list, so they cannot drift
 * apart the way a second hand-maintained classification would.
 *
 * Two categories, matching the sidebar experiment:
 *   - 'standard': `type: standard`, and `type: report` when it carries a
 *     document number (ECAC Doc 29, Directive 2003/10/EC and friends are
 *     normative documents in everything but the Zod tag).
 *   - 'reference': `type: article` and `type: book`, the named work the method
 *     is attributed to, shown as author and year.
 * `type: web` and numberless reports carry no designation and no citable
 * author-date pair, so they stay in the bibliography only.
 */

import type { CollectionEntry } from 'astro:content';

export type Reference = NonNullable<CollectionEntry<'docs'>['data']['references']>[number];

export type ChipKind = 'standard' | 'reference';

export interface Chip {
	/** Short label: standard family, or author and year. */
	text: string;
	kind: ChipKind;
	/** Fragment id of the matching bibliography entry, without the '#'. */
	anchor: string;
}

/**
 * Caps for the header run. Five standards plus three named works is the most
 * that still fits on one line at a desktop reading width, and one line is
 * what makes the run read as a dateline instead of a block. Anything past the
 * caps folds into a single "+N more" link to the References section, which is
 * the honest answer for a literature-heavy page: the bibliography is right
 * there.
 *
 * Which entries survive the cap is deterministic: entries flagged
 * `primary: true` in the frontmatter come first, everything else keeps
 * frontmatter order. A page whose bibliography is longer than the caps should
 * flag the sources it actually leans on.
 */
export const MAX_STANDARDS = 5;
export const MAX_REFERENCES = 3;

function slug(value: string): string {
	return value
		.normalize('NFD')
		.replace(/[\u0300-\u036f]/g, '')
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '-')
		.replace(/^-+|-+$/g, '');
}

/** Surname out of an APA-formatted name ("Schroeder, M. R." -> "Schroeder"). */
function surname(author: string): string {
	return author.split(',')[0]!.trim();
}

/**
 * Designation reduced to the family that a reader recognises:
 * "ISO 10140-2:2010" -> "ISO 10140", "ANSI S3.5-1997 (R2017)" -> "ANSI S3.5",
 * "ECAC.CEAC Doc 29, 4th ed." -> "ECAC.CEAC Doc 29". Parts of one standard
 * collapse onto a single chip; distinct standards keep distinct chips.
 */
export function standardFamily(designation: string): string {
	let text = designation.split(':')[0]!.split(',')[0]!.split(' (')[0]!.split('+')[0]!.trim();
	text = text.replace(/^Recommendation\s+/i, '');
	// Trailing edition year first ("ANSI/ASA S12.2-2019"), then part number
	// ("ISO 10140-2"). A three-digit or longer tail is part of the number
	// itself ("NIOSH Publication No. 98-126") and stays.
	text = text.replace(/-\d{4}$/, '');
	text = text.replace(/-\d{1,2}$/, '');
	return text.trim();
}

/** An "8th ed." style designation names an edition, not a document. */
function isEditionOnly(text: string): boolean {
	return /^\d+\s*(st|nd|rd|th|\.ª|ª|a)?\s*ed\.?$/i.test(text) || !/\d/.test(text);
}

function authorDate(
	authors: readonly string[] | undefined,
	year: number | string | undefined,
	isEs: boolean,
): string | undefined {
	if (!authors?.length) return undefined;
	const names =
		authors.length === 1
			? surname(authors[0]!)
			: authors.length === 2
				? `${surname(authors[0]!)} ${isEs ? 'y' : '&'} ${surname(authors[1]!)}`
				: `${surname(authors[0]!)} et al.`;
	return year === undefined ? names : `${names} ${year}`;
}

function chipFor(ref: Reference, isEs: boolean): { text: string; kind: ChipKind } | undefined {
	if (ref.type === 'standard') {
		const text = standardFamily(ref.designation);
		return isEditionOnly(text) ? undefined : { text, kind: 'standard' };
	}
	if (ref.type === 'report' && ref.number) {
		const text = standardFamily(ref.number);
		return isEditionOnly(text) ? undefined : { text, kind: 'standard' };
	}
	if (ref.type === 'article' || ref.type === 'book') {
		const text = authorDate(ref.authors, ref.year, isEs);
		return text ? { text, kind: 'reference' } : undefined;
	}
	return undefined;
}

/**
 * Fragment ids for the bibliography entries, parallel to the input array so
 * both components address the same entry by position regardless of the order
 * they render in. Repeated keys get a numeric suffix.
 */
export function referenceAnchors(references: readonly Reference[]): string[] {
	const seen = new Map<string, number>();
	return references.map((ref) => {
		const base =
			ref.type === 'standard'
				? slug(ref.designation)
				: ref.type === 'report' && ref.number
					? slug(ref.number)
					: slug(
							[
								'authors' in ref && ref.authors?.length
									? surname(ref.authors[0]!)
									: ('organization' in ref && ref.organization) || ref.title,
								ref.year ?? '',
							].join(' '),
						);
		const count = (seen.get(base) ?? 0) + 1;
		seen.set(base, count);
		return `ref-${base}${count > 1 ? `-${count}` : ''}`;
	});
}

export interface PageChips {
	standards: Chip[];
	references: Chip[];
	/** Entries that carry a chip but did not fit under the caps. */
	hidden: number;
}

/**
 * Header chips for one page, standards first, each pointing at its entry in
 * the page bibliography. Duplicated labels (parts of the same standard, two
 * editions of the same report) collapse onto the first occurrence.
 */
export function pageChips(references: readonly Reference[] | undefined, isEs: boolean): PageChips {
	const empty: PageChips = { standards: [], references: [], hidden: 0 };
	if (!references?.length) return empty;

	const anchors = referenceAnchors(references);
	const standards: Chip[] = [];
	const named: Chip[] = [];
	const seen = new Set<string>();

	// Flagged entries first, frontmatter order within each half. The sort is
	// stable, so an unflagged bibliography behaves exactly as before.
	const ordered = [...references.entries()].sort(
		([, a], [, b]) => Number(b.primary ?? false) - Number(a.primary ?? false),
	);
	for (const [index, ref] of ordered) {
		const chip = chipFor(ref, isEs);
		if (!chip || seen.has(chip.text)) continue;
		seen.add(chip.text);
		(chip.kind === 'standard' ? standards : named).push({ ...chip, anchor: anchors[index]! });
	}

	const shownStandards = Math.min(standards.length, MAX_STANDARDS);
	const shownReferences = Math.min(named.length, MAX_REFERENCES);
	return {
		standards: standards.slice(0, shownStandards),
		references: named.slice(0, shownReferences),
		hidden: standards.length - shownStandards + (named.length - shownReferences),
	};
}
