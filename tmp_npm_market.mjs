const names = ["dsh-market", "@dsh/market", "dsh-plugin-market", "dsh-market-plugin"];

for (const name of names) {
  const url = "https://registry.npmjs.org/" + name.replace("/", "%2F");
  try {
    const r = await fetch(url);
    if (!r.ok) { console.log(`${name}: HTTP ${r.status}`); continue; }
    const j = await r.json();
    const latest = j["dist-tags"]?.latest;
    const v = j.versions?.[latest];
    console.log(`${name}: OK latest=${latest} versions=${Object.keys(j.versions).length} modified=${j.time?.modified}`);
    if (v) {
      console.log("   tarball:", v.dist?.tarball);
      console.log("   lifecycle:", JSON.stringify(Object.entries(v.scripts ?? {}).filter(([k]) => ["preinstall", "install", "postinstall", "prepare", "prepack"].includes(k))));
      console.log("   engines:", JSON.stringify(v.engines), "deps:", JSON.stringify(Object.keys(v.dependencies ?? {})));
    }
  } catch (e) {
    console.log(`${name}: ERR ${e.message}`);
  }
}

try {
  const r = await fetch("https://registry.npmjs.org/-/v1/search?text=dsh-market&size=10");
  const j = await r.json();
  console.log("\nsearch 'dsh-market':");
  for (const o of j.objects ?? []) console.log(`  ${o.package.name}@${o.package.version} — ${o.package.description ?? ""}`.slice(0, 160));
} catch (e) { console.log("search ERR", e.message); }
