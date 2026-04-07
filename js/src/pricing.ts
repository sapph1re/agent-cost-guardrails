export interface ModelPrice {
  inputPerMtok: number;  // USD per 1M input tokens
  outputPerMtok: number; // USD per 1M output tokens
}

export type CustomPricing = Record<string, { input_per_mtok: number; output_per_mtok: number }>;

// Bundled pricing table — covers major models.
// Prices are per 1M tokens in USD.
const PRICING: Record<string, ModelPrice> = {
  // OpenAI
  'gpt-4o': { inputPerMtok: 2.50, outputPerMtok: 10.00 },
  'gpt-4o-mini': { inputPerMtok: 0.15, outputPerMtok: 0.60 },
  'gpt-4-turbo': { inputPerMtok: 10.00, outputPerMtok: 30.00 },
  'gpt-4': { inputPerMtok: 30.00, outputPerMtok: 60.00 },
  'gpt-3.5-turbo': { inputPerMtok: 0.50, outputPerMtok: 1.50 },
  'o1': { inputPerMtok: 15.00, outputPerMtok: 60.00 },
  'o1-mini': { inputPerMtok: 3.00, outputPerMtok: 12.00 },
  'o1-pro': { inputPerMtok: 150.00, outputPerMtok: 600.00 },
  'o3-mini': { inputPerMtok: 1.10, outputPerMtok: 4.40 },
  'gpt-4.1': { inputPerMtok: 2.00, outputPerMtok: 8.00 },
  'gpt-4.1-mini': { inputPerMtok: 0.40, outputPerMtok: 1.60 },
  'gpt-4.1-nano': { inputPerMtok: 0.10, outputPerMtok: 0.40 },
  // Anthropic
  'claude-opus-4-6': { inputPerMtok: 15.00, outputPerMtok: 75.00 },
  'claude-sonnet-4-6': { inputPerMtok: 3.00, outputPerMtok: 15.00 },
  'claude-haiku-4-5-20251001': { inputPerMtok: 0.80, outputPerMtok: 4.00 },
  'claude-3-5-sonnet-20241022': { inputPerMtok: 3.00, outputPerMtok: 15.00 },
  'claude-3-5-haiku-20241022': { inputPerMtok: 0.80, outputPerMtok: 4.00 },
  'claude-3-opus-20240229': { inputPerMtok: 15.00, outputPerMtok: 75.00 },
  'claude-3-sonnet-20240229': { inputPerMtok: 3.00, outputPerMtok: 15.00 },
  'claude-3-haiku-20240307': { inputPerMtok: 0.25, outputPerMtok: 1.25 },
  // Google
  'gemini-2.0-flash': { inputPerMtok: 0.10, outputPerMtok: 0.40 },
  'gemini-2.0-pro': { inputPerMtok: 1.25, outputPerMtok: 10.00 },
  'gemini-1.5-pro': { inputPerMtok: 1.25, outputPerMtok: 5.00 },
  'gemini-1.5-flash': { inputPerMtok: 0.075, outputPerMtok: 0.30 },
  // Mistral
  'mistral-large-latest': { inputPerMtok: 2.00, outputPerMtok: 6.00 },
  'mistral-small-latest': { inputPerMtok: 0.10, outputPerMtok: 0.30 },
  // DeepSeek
  'deepseek-chat': { inputPerMtok: 0.27, outputPerMtok: 1.10 },
  'deepseek-reasoner': { inputPerMtok: 0.55, outputPerMtok: 2.19 },
  // Meta (via providers)
  'llama-3.1-405b': { inputPerMtok: 3.00, outputPerMtok: 3.00 },
  'llama-3.1-70b': { inputPerMtok: 0.88, outputPerMtok: 0.88 },
  'llama-3.1-8b': { inputPerMtok: 0.18, outputPerMtok: 0.18 },
};

let customPricing: Record<string, ModelPrice> = {};

export function setCustomPricing(prices: CustomPricing): void {
  for (const [model, rates] of Object.entries(prices)) {
    customPricing[model] = {
      inputPerMtok: rates.input_per_mtok,
      outputPerMtok: rates.output_per_mtok,
    };
  }
}

export function clearCustomPricing(): void {
  customPricing = {};
}

export function getModelPrice(model: string): ModelPrice | null {
  if (customPricing[model]) return customPricing[model];
  if (PRICING[model]) return PRICING[model];
  // Prefix matching (e.g., "gpt-4o-2024-05-13" -> "gpt-4o")
  const keys = Object.keys(PRICING).sort((a, b) => b.length - a.length);
  for (const key of keys) {
    if (model.startsWith(key)) return PRICING[key];
  }
  return null;
}

export function calculateCost(
  model: string,
  inputTokens: number,
  outputTokens: number,
  custom?: CustomPricing,
): number {
  let price: ModelPrice | null = null;
  if (custom && custom[model]) {
    price = { inputPerMtok: custom[model].input_per_mtok, outputPerMtok: custom[model].output_per_mtok };
  } else {
    price = getModelPrice(model);
  }
  if (!price) return 0;
  return (inputTokens * price.inputPerMtok / 1_000_000) + (outputTokens * price.outputPerMtok / 1_000_000);
}
