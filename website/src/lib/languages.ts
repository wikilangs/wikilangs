/**
 * Language data utilities for the Wikilangs website.
 *
 * Language data is fetched at build time by scripts/fetch_languages.py
 * and stored in src/data/languages.json
 */

import languagesData from '../data/languages.json';

export interface LanguageData {
  code: string;
  name: string;
  common_name: string | null;
  native_name: string | null;  // Language name in its own script
  text_direction: 'ltr' | 'rtl';  // Text direction from Wikipedia
  alpha_2: string | null;
  alpha_3: string | null;
  scope: string | null;
  language_type: string | null;

  // Metrics
  vocabulary_size: number | null;
  best_compression_ratio: number | null;
  best_isotropy: number | null;

  // Content
  has_models: boolean;
  model_card_excerpt: string | null;
  wikipedia_samples: string[];  // Pre-fetched samples from Wikipedia

  // URLs
  hf_url: string;
  visualizations_base: string;
}

// Type assertion for imported JSON
const languages = languagesData as LanguageData[];

/**
 * Get all languages
 */
export function getAllLanguages(): LanguageData[] {
  return languages;
}

/**
 * Get languages that have models available
 */
export function getLanguagesWithModels(): LanguageData[] {
  return languages.filter(lang => lang.has_models);
}

/**
 * Get a single language by code
 */
export function getLanguage(code: string): LanguageData | undefined {
  return languages.find(lang => lang.code === code);
}

/**
 * Get languages grouped by first letter (for alphabetical navigation)
 */
export function getLanguagesByLetter(): Map<string, LanguageData[]> {
  const byLetter = new Map<string, LanguageData[]>();

  for (const lang of languages) {
    const letter = lang.name[0].toUpperCase();
    if (!byLetter.has(letter)) {
      byLetter.set(letter, []);
    }
    byLetter.get(letter)!.push(lang);
  }

  return byLetter;
}

/**
 * Get global statistics across all languages
 */
export function getGlobalStats() {
  const withModels = languages.filter(l => l.has_models);
  const withMetrics = languages.filter(l => l.vocabulary_size !== null);

  const totalVocab = withMetrics.reduce((sum, l) => sum + (l.vocabulary_size || 0), 0);
  const avgCompression = withMetrics.reduce((sum, l) => sum + (l.best_compression_ratio || 0), 0) / withMetrics.length;

  return {
    totalLanguages: languages.length,
    languagesWithModels: withModels.length,
    languagesWithMetrics: withMetrics.length,
    totalVocabularyWords: totalVocab,
    averageCompressionRatio: avgCompression,
  };
}

/**
 * Get visualization URL for a language
 */
export function getVisualizationUrl(langCode: string, vizName: string): string {
  return `https://huggingface.co/wikilangs/${langCode}/resolve/main/visualizations/${vizName}.png`;
}

/**
 * Format a number with locale-aware separators
 */
export function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—';
  return n.toLocaleString('en-US');
}

/**
 * Format compression ratio
 */
export function formatCompression(ratio: number | null | undefined): string {
  if (ratio === null || ratio === undefined) return '—';
  return `${ratio.toFixed(2)}x`;
}
