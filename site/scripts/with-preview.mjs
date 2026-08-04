// Runs a command against `astro preview`, and only once the preview answers.
//
// Both audits used to start the preview in the background, sleep three seconds
// and hope. Three seconds is generous on this machine and a guess on a shared
// CI worker: the audit that loses the race fails on a connection refused, which
// reads like a broken page and is not one. This polls the origin instead, so
// the wait is as long as it needs to be and no longer, and it stops the preview
// on every exit path rather than only the ordinary one.
//
// Usage: node scripts/with-preview.mjs [--port 4321] -- <command...>
import { spawn } from 'node:child_process';

const argv = process.argv.slice(2);
const split = argv.indexOf('--');
if (split === -1) {
  console.error('with-preview: no command. Usage: with-preview.mjs [--port N] -- <command...>');
  process.exit(2);
}
const flags = argv.slice(0, split);
const command = argv.slice(split + 1);
const portFlag = flags.indexOf('--port');
const port = portFlag === -1 ? '4321' : flags[portFlag + 1];

// Two minutes: an Astro preview is serving a built tree, so it is up in
// milliseconds here and in seconds on a cold worker. A ceiling this high never
// fires in practice and turns a hung start into a clear message rather than a
// job that runs to the six-hour limit.
const READY_TIMEOUT_MS = 120_000;
const POLL_MS = 300;

// Something already on the port is the trap this runner exists to close, not
// a convenience: `astro preview` would fail to bind and exit, the poll below
// would be answered by whatever is there, and the audit would report on a
// different site. A dev server left open on the default port is exactly how
// that happens.
try {
  await fetch(`http://localhost:${port}/`, { signal: AbortSignal.timeout(2000) });
  console.error(
    `with-preview: something is already serving on port ${port}. Stop it, or ` +
      `pass --port with a free one, so the audit reads this build and not that.`,
  );
  process.exit(1);
} catch {
  // Nothing listening, which is what we want.
}

const preview = spawn('./node_modules/.bin/astro', ['preview', '--port', port], {
  stdio: ['ignore', 'inherit', 'inherit'],
});

let stopped = false;
const stop = () => {
  if (stopped) return;
  stopped = true;
  preview.kill('SIGTERM');
};
for (const signal of ['exit', 'SIGINT', 'SIGTERM', 'uncaughtException']) {
  process.on(signal, stop);
}

/** Resolves once the origin answers anything at all, or throws on the ceiling. */
async function waitForReady(origin) {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  for (;;) {
    if (preview.exitCode !== null) {
      throw new Error(`astro preview exited with ${preview.exitCode} before serving`);
    }
    try {
      // Any status is proof the server is listening: a 404 on the origin is a
      // served response, and the base path is the audit's business, not ours.
      await fetch(origin, { signal: AbortSignal.timeout(2000) });
      return;
    } catch {
      if (Date.now() > deadline) {
        throw new Error(`astro preview did not answer on ${origin} within ${READY_TIMEOUT_MS} ms`);
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    }
  }
}

try {
  await waitForReady(`http://localhost:${port}/`);
} catch (error) {
  console.error(`with-preview: ${error.message}`);
  stop();
  process.exit(1);
}

const child = spawn(command[0], command.slice(1), { stdio: 'inherit', shell: false });
child.on('exit', (code, signal) => {
  stop();
  process.exit(signal ? 1 : (code ?? 1));
});
child.on('error', (error) => {
  console.error(`with-preview: ${error.message}`);
  stop();
  process.exit(1);
});
