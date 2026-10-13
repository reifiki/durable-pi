import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Store } from "../src/store.js";
import { loadConfig } from "../src/config.js";

// The NVIDIA catalog is static and shipped with pi-ai, so these need no network
// and no API key: only what is reachable through Store is asserted.
// Model availability is gated on the presence of the key, not on calling the API,
// so a placeholder is enough to exercise the catalog offline.
process.env.NVIDIA_API_KEY ||= "test-key-not-used";

const temp = async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "dpi-nv-")));
  return {
    root,
    options: { dataDir: join(root, "data"), startDirs: [root], name: "nv", provider: "nvidia" as const },
    cleanup: () => rm(root, { recursive: true, force: true }),
  };
};

test("nvidia provider offers only zero-cost models and defaults to a free one", async () => {
  const t = await temp();
  const store = await Store.open(t.options);
  try {
    const models = store.modelList;
    assert.ok(models.length > 0);
    for (const m of models) {
      assert.equal(m.provider, "nvidia");
      assert.deepEqual(m.cost, { input: 0, output: 0 });
    }
    const fallback = store.routedModel(undefined);
    assert.equal(fallback.provider, "nvidia");
    assert.ok(models.some((m) => m.id === fallback.modelId), `default ${fallback.modelId} not offered`);
  } finally {
    await store.close();
    await t.cleanup();
  }
});

test("nvidia rejects paid models and unknown ids, and honours DPI_DEFAULT_MODEL", async () => {
  const t = await temp();
  const store = await Store.open({ ...t.options, defaultModel: "z-ai/glm-5.3-flash" });
  try {
    assert.deepEqual(store.routedModel(undefined), { provider: "nvidia", modelId: "z-ai/glm-5.3-flash" });
    // A paid model from the same catalog is not a choice on this provider.
    assert.throws(
      () => store.validateModel({ provider: "nvidia", modelId: "nvidia/nemotron-3-ultra-550b-a55b" }),
      /Unknown model/,
    );
    assert.throws(() => store.validateModel({ provider: "nvidia", modelId: "made-up/model" }), /Unknown model/);
    // An unknown DPI_DEFAULT_MODEL fails loudly instead of silently falling back.
    await assert.rejects(
      () => Store.open({ ...t.options, dataDir: join(t.root, "data2"), defaultModel: "made-up/model" }),
      /not a free NVIDIA model/,
    );
    // A model from another provider never leaks through routing.
    assert.equal(store.routedModel({ provider: "openrouter", modelId: "anthropic/claude-sonnet-5.5" }).provider, "nvidia");
  } finally {
    await store.close();
    await t.cleanup();
  }
});

test("config accepts the nvidia provider and rejects typos", () => {
  const env = { DPI_PROVIDER: "nvidia", DPI_DEFAULT_MODEL: "z-ai/glm-5.3-flash", DPI_TOKEN: "x".repeat(40) } as Record<
    string,
    string
  >;
  const config = loadConfig([], { ...env, DPI_ENV_FILE: join(tmpdir(), "dpi-nonexistent.env") });
  assert.equal(config.provider, "nvidia");
  assert.equal(config.defaultModel, "z-ai/glm-5.3-flash");
  assert.throws(
    () => loadConfig([], { ...env, DPI_PROVIDER: "opencde", DPI_ENV_FILE: env.DPI_ENV_FILE }),
    /Unknown DPI_PROVIDER/,
  );
});