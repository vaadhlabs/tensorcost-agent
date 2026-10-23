#!/usr/bin/env node
/**
 * Fail when a service migration touches another service's Postgres schema without
 * a preceding existence guard in the same DO block (boot blockers for optional verticals).
 *
 * Scans DO $$ blocks for REFERENCES, GRANT, FROM, and JOIN.
 *
 * Usage:
 *   node check-cross-schema-migrations.core.mjs --root /path/to/repo --schema-map ./map.json
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
      console.log(
        "Usage: node check-cross-schema-migrations.core.mjs --root <repo> --schema-map <json>",
      );
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
const SCHEMA_OWNER = Object.fromEntries(
  Object.entries(SCHEMA_BY_SERVICE).map(([svc, schema]) => [schema, svc]),
);

if (!SCHEMA_BY_SERVICE || typeof SCHEMA_BY_SERVICE !== "object") {
  console.error("schema-map must include schemaByService object");
  process.exit(2);
}

function stripComments(sql) {
  return sql.replace(/--[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "");
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

/** @returns {{ start: number, end: number, body: string }[]} */
function extractDoBlocks(sql) {
  const blocks = [];
  const re = /DO\s+\$\$([\s\S]*?)END\s*\$\$\s*;/gi;
  let m;
  while ((m = re.exec(sql)) !== null) {
    const body = m[1] ?? "";
    const bodyStart = m.index + m[0].indexOf(body);
    blocks.push({
      start: m.index,
      end: m.index + m[0].length,
      bodyStart,
      body,
    });
  }
  return blocks;
}

function isInsideDoBlock(offset, blocks) {
  return blocks.find((b) => offset >= b.start && offset < b.end);
}

const CROSS_SCHEMA_PATTERNS = [
  { name: "REFERENCES", re: /REFERENCES\s+"?([a-z][a-z0-9_]*)"?\./gi },
  { name: "GRANT_ON", re: /\bGRANT\b[\s\S]{0,120}?\bON\s+"?([a-z][a-z0-9_]*)"?\./gi },
  { name: "GRANT_SCHEMA", re: /GRANT\s+USAGE\s+ON\s+SCHEMA\s+"?([a-z][a-z0-9_]*)"?/gi },
  { name: "FROM", re: /\bFROM\s+"?([a-z][a-z0-9_]*)"?\./gi },
  { name: "JOIN", re: /\bJOIN\s+"?([a-z][a-z0-9_]*)"?\./gi },
];

function isForeignSchema(schema, ownSchema) {
  if (schema === ownSchema || schema === "public" || schema === "platform") return false;
  return OPTIONAL_VERTICAL_SCHEMAS.has(schema) || schema === "cost";
}

const SCHEMA_GUARD_RE = (schema) =>
  new RegExp(`schema_name\\s*=\\s*'${schema}'|table_schema\\s*=\\s*'${schema}'`, "i");

/** Pattern A: IF EXISTS (…schema…) THEN … <cross-schema stmt> (cost-service 020 style). */
function stmtInsideIfExistsForSchema(prefix, schema, offsetInBlock) {
  const guardRe = new RegExp(
    `schema_name\\s*=\\s*'${schema}'|table_schema\\s*=\\s*'${schema}'`,
    "gi",
  );
  let guardMatch;
  let lastGuardIdx = -1;
  while ((guardMatch = guardRe.exec(prefix)) !== null) {
    lastGuardIdx = guardMatch.index;
  }
  if (lastGuardIdx === -1) return false;

  const windowStart = Math.max(0, lastGuardIdx - 400);
  const guardWindow = prefix.slice(windowStart, lastGuardIdx + 80);
  if (!/IF\s+EXISTS/i.test(guardWindow)) return false;

  const afterGuard = prefix.slice(lastGuardIdx);
  const thenMatch = /\bTHEN\b/i.exec(afterGuard);
  if (!thenMatch) return false;
  const thenAbs = lastGuardIdx + thenMatch.index;
  const betweenThenAndStmt = prefix.slice(thenAbs, offsetInBlock);
  if (/\bEND\s+IF\b/i.test(betweenThenAndStmt)) return false;
  return true;
}

/** Pattern B: IF NOT EXISTS (…schema…) THEN RETURN; END IF; … <stmt>. */
function stmtAfterSchemaNotExistsEarlyReturn(prefix, schema) {
  const endIfRe = /\bEND\s+IF\b/gi;
  let lastEndIf = -1;
  let m;
  while ((m = endIfRe.exec(prefix)) !== null) {
    lastEndIf = m.index;
  }
  if (lastEndIf === -1) return false;

  const guardedPrefix = prefix.slice(0, lastEndIf);
  if (!/IF\s+NOT\s+EXISTS/i.test(guardedPrefix)) return false;
  if (!SCHEMA_GUARD_RE(schema).test(guardedPrefix)) return false;
  const tail = prefix.slice(Math.max(0, lastEndIf - 120), lastEndIf);
  return /\bRETURN\s*;/i.test(tail);
}

function crossSchemaStmtInsideSchemaGuardIf(blockBody, schema, offsetInBlock) {
  const prefix = blockBody.slice(0, offsetInBlock);
  return (
    stmtInsideIfExistsForSchema(prefix, schema, offsetInBlock) ||
    stmtAfterSchemaNotExistsEarlyReturn(prefix, schema)
  );
}

function hasLedgerDefer(sql) {
  return /@ledger-deferred:/i.test(sql.split("\n").slice(0, 5).join("\n"));
}

function ownerHasJobsGrantMigration(ownerSvc) {
  const ownerDir = join(SERVICES_DIR, ownerSvc, "migrations");
  try {
    return readdirSync(ownerDir).some(
      (f) =>
        f.includes("jobs_db_role") ||
        f.includes("jobs_role_grants") ||
        f.includes("tensorcost_jobs"),
    );
  } catch {
    return false;
  }
}

const violations = [];

for (const [svc, ownSchema] of Object.entries(SCHEMA_BY_SERVICE)) {
  const svcDir = join(SERVICES_DIR, svc);
  if (!statSync(svcDir, { throwIfNoEntry: false })?.isDirectory()) continue;

  for (const file of listMigrationFiles(svcDir)) {
    const raw = readFileSync(file, "utf8");
    const sql = stripComments(raw);
    const doBlocks = extractDoBlocks(sql);
    const rel = relative(ROOT, file);

    for (const { name, re } of CROSS_SCHEMA_PATTERNS) {
      re.lastIndex = 0;
      let m;
      while ((m = re.exec(sql)) !== null) {
        const schema = m[1];
        if (!isForeignSchema(schema, ownSchema)) continue;

        const block = isInsideDoBlock(m.index, doBlocks);
        if (block) {
          const offsetInBody = m.index - block.bodyStart;
          if (crossSchemaStmtInsideSchemaGuardIf(block.body, schema, offsetInBody)) {
            continue;
          }
          violations.push({
            file: rel,
            schema,
            kind: name,
            detail:
              "DO block: cross-schema statement not inside IF NOT EXISTS guard for that schema",
          });
        } else {
          violations.push({
            file: rel,
            schema,
            kind: name,
            detail: "unconditional cross-schema (outside DO)",
          });
        }
      }
    }

    if (svc === "cost-service" && /tensorcost_jobs/.test(sql)) {
      for (const schema of ["ai", "alert", "audit", "gpu"]) {
        if (!sql.includes(`'${schema}'`) && !sql.includes(`"${schema}"`)) continue;
        const owner = SCHEMA_OWNER[schema];
        if (!owner || owner === "cost-service") continue;
        if (!ownerHasJobsGrantMigration(owner) && !hasLedgerDefer(raw)) {
          violations.push({
            file: rel,
            schema,
            kind: "DEFERRED_GRANT",
            detail: `cost-service grants to ${schema} need companion migration in ${owner}`,
          });
        }
      }
    }
  }
}

if (violations.length > 0) {
  console.error("Cross-schema migration violations:\n");
  for (const v of violations) {
    console.error(`  ${v.file}: ${v.kind} on "${v.schema}" — ${v.detail}`);
  }
  process.exit(1);
}

console.log("check-cross-schema-migrations: ok");
