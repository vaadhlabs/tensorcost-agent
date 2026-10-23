#!/usr/bin/env node
/**
 * Fail when a service migration unconditionally REFERENCES another service's
 * Postgres schema (boot blockers for optional verticals / modular monoliths).
 *
 * Usage:
 *   node check-cross-schema-migrations.mjs --root /path/to/repo --schema-map ./map.json
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

function parseArgs(argv) {
  const out = { root: "", schemaMap: "" };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--root" && argv[i + 1]) {
      out.root = argv[++i];
    } else if (a === "--schema-map" && argv[i + 1]) {
      out.schemaMap = argv[++i];
    } else if (a === "-h" || a === "--help") {
      console.log(`Usage: node check-cross-schema-migrations.mjs --root <repo> --schema-map <json>`);
      process.exit(0);
    }
  }
  return out;
}

const { root: rootArg, schemaMap: mapArg } = parseArgs(process.argv);
if (!rootArg || !mapArg) {
  console.error("Required: --root and --schema-map");
  process.exit(2);
}

const ROOT = resolve(rootArg);
const map = JSON.parse(readFileSync(resolve(mapArg), "utf8"));
const SERVICES_DIR = join(ROOT, map.servicesDir ?? "backend/services");
const SCHEMA_BY_SERVICE = map.schemaByService;
const OPTIONAL_VERTICAL_SCHEMAS = new Set(map.optionalVerticalSchemas ?? []);

if (!SCHEMA_BY_SERVICE || typeof SCHEMA_BY_SERVICE !== "object") {
  console.error("schema-map must include schemaByService object");
  process.exit(2);
}

function stripComments(sql) {
  return sql.replace(/--[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "");
}

function stripDoBlocks(sql) {
  return sql.replace(/DO\s+\$\$[\s\S]*?END\s*\$\$\s*;/gi, "");
}

function listMigrationFiles(serviceDir) {
  const migDir = join(serviceDir, "migrations");
  try {
    return readdirSync(migDir)
      .filter((f) => f.endsWith(".sql"))
      .map((f) => join(migDir, f));
  } catch {
    return [];
  }
}

const violations = [];
const refRe = /REFERENCES\s+"?([a-z][a-z0-9_]*)"?\./gi;

for (const [svc, ownSchema] of Object.entries(SCHEMA_BY_SERVICE)) {
  const svcDir = join(SERVICES_DIR, svc);
  if (!statSync(svcDir, { throwIfNoEntry: false })?.isDirectory()) continue;

  for (const file of listMigrationFiles(svcDir)) {
    const body = stripDoBlocks(stripComments(readFileSync(file, "utf8")));

    let m;
    refRe.lastIndex = 0;
    while ((m = refRe.exec(body)) !== null) {
      const schema = m[1];
      if (schema === ownSchema || schema === "public" || schema === "platform") continue;
      if (!OPTIONAL_VERTICAL_SCHEMAS.has(schema) && schema !== "cost") continue;
      violations.push({ file: relative(ROOT, file), schema });
    }
  }
}

if (violations.length > 0) {
  console.error("Unconditional cross-schema REFERENCES:\n");
  for (const v of violations) {
    console.error(`  ${v.file}: REFERENCES "${v.schema}"`);
  }
  process.exit(1);
}

console.log("check-cross-schema-migrations: ok");
