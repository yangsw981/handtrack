const r = await fetch("https://registry.npmjs.org/dsh-ops");
if (!r.ok) {
  console.log("registry status", r.status);
  process.exit(1);
}
const j = await r.json();
console.log("name:", j.name);
console.log("dist-tags:", JSON.stringify(j["dist-tags"]));
const versions = Object.keys(j.versions);
console.log("versions:", versions.slice(-8).join(", "));
const latest = j["dist-tags"]?.latest;
const v = j.versions[latest];
if (v) {
  console.log("latest:", latest, "tarball:", v.dist.tarball);
  console.log("lifecycle scripts:", JSON.stringify(Object.keys(v.scripts ?? {}).filter((s) => ["preinstall", "install", "postinstall", "prepare", "prepack"].includes(s)).map((s) => s + "=" + v.scripts[s])));
  console.log("optionalDependencies:", JSON.stringify(v.optionalDependencies));
  console.log("engines:", JSON.stringify(v.engines));
}
