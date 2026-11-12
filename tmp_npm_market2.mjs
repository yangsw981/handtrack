const names = ["@dsh-market/plugin", "dshmarket", "safer-dsh-market", "dsh-market-plus"];

for (const name of names) {
  const enc = name.replace("/", "%2F");
  try {
    const r = await fetch("https://registry.npmjs.org/" + enc);
    if (!r.ok) { console.log(`${name}: HTTP ${r.status}\n`); continue; }
    const j = await r.json();
    const latest = j["dist-tags"]?.latest;
    const v = j.versions[latest];
    const dl = await fetch(`https://api.npmjs.org/downloads/point/last-week/${enc}`).then((x) => x.json()).catch(() => ({}));
    console.log("=== " + name + " @ " + latest);
    console.log("  desc:", (v.description ?? "").slice(0, 200));
    console.log("  published:", j.time?.[latest]);
    console.log("  downloads(last-week):", dl.downloads ?? "n/a");
    console.log("  repo:", typeof v.repository === "string" ? v.repository : v.repository?.url);
    console.log("  engines:", JSON.stringify(v.engines));
    console.log("  dsh field:", JSON.stringify(v.dsh ?? null).slice(0, 300));
    console.log("  lifecycle:", JSON.stringify(Object.entries(v.scripts ?? {}).filter(([k]) => ["preinstall", "install", "postinstall", "prepare", "prepack"].includes(k))));
    console.log("  deps:", Object.keys(v.dependencies ?? {}).length, "peer:", Object.keys(v.peerDependencies ?? {}).join(","));
    console.log("  optionalDeps:", Object.keys(v.optionalDependencies ?? {}).join(","));
    console.log();
  } catch (e) { console.log(name, "ERR", e.message); }
}
