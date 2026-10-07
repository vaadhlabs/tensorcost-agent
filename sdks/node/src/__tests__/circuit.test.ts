/**
 * Tests for the circuit breaker state machine.
 */

import { describe, expect, it } from "vitest";
import { CircuitBreaker } from "../circuit.js";

function makeCircuit(openThreshold = 3, closeThreshold = 5) {
  return new CircuitBreaker({ openThreshold, closeThreshold });
}

describe("CircuitBreaker", () => {
  it("starts CLOSED and does not bypass", () => {
    const cb = makeCircuit();
    expect(cb.currentState).toBe("CLOSED");
    expect(cb.shouldBypass()).toBe(false);
    expect(cb.failures).toBe(0);
  });

  it("opens after openThreshold consecutive failures", () => {
    const cb = makeCircuit(3, 5);
    cb.recordFailure();
    expect(cb.currentState).toBe("CLOSED");
    cb.recordFailure();
    expect(cb.currentState).toBe("CLOSED");
    cb.recordFailure(); // 3rd failure
    expect(cb.currentState).toBe("OPEN");
    expect(cb.shouldBypass()).toBe(true);
  });

  it("resets failure count on success while CLOSED", () => {
    const cb = makeCircuit(3, 5);
    cb.recordFailure();
    cb.recordFailure();
    cb.recordSuccess();
    expect(cb.failures).toBe(0);
    // Two more failures should not open (count reset)
    cb.recordFailure();
    cb.recordFailure();
    expect(cb.currentState).toBe("CLOSED");
  });

  it("transitions OPEN -> HALF_OPEN via shouldProbe()", () => {
    const cb = makeCircuit(2, 2);
    cb.recordFailure();
    cb.recordFailure();
    expect(cb.currentState).toBe("OPEN");
    expect(cb.shouldProbe()).toBe(true);
    expect(cb.currentState).toBe("HALF_OPEN");
  });

  it("HALF_OPEN -> CLOSED after closeThreshold probe successes", () => {
    const cb = makeCircuit(2, 3);
    cb.recordFailure();
    cb.recordFailure();
    cb.shouldProbe(); // OPEN -> HALF_OPEN
    cb.recordSuccess();
    cb.recordSuccess();
    expect(cb.currentState).toBe("HALF_OPEN"); // not yet
    cb.recordSuccess(); // 3rd success
    expect(cb.currentState).toBe("CLOSED");
    expect(cb.failures).toBe(0);
  });

  it("HALF_OPEN -> OPEN on probe failure", () => {
    const cb = makeCircuit(2, 5);
    cb.recordFailure();
    cb.recordFailure();
    cb.shouldProbe(); // -> HALF_OPEN
    cb.recordSuccess();
    cb.recordSuccess();
    cb.recordFailure(); // probe failed -> back to OPEN
    expect(cb.currentState).toBe("OPEN");
  });

  it("shouldProbe returns false when CLOSED", () => {
    const cb = makeCircuit();
    expect(cb.shouldProbe()).toBe(false);
    expect(cb.currentState).toBe("CLOSED");
  });

  it("single-flight HALF_OPEN probe — concurrent OPEN callers bypass", () => {
    const cb = makeCircuit(1, 5);
    cb.recordFailure();
    expect(cb.currentState).toBe("OPEN");
    expect(cb.shouldProbe()).toBe(true);
    expect(cb.currentState).toBe("HALF_OPEN");
    expect(cb.shouldBypass()).toBe(true);
    expect(cb.shouldProbe()).toBe(false);
  });

  it("clears probeInFlight after successful probe so recovery continues", () => {
    const cb = makeCircuit(1, 3);
    cb.recordFailure();
    cb.shouldProbe();
    cb.recordSuccess();
    expect(cb.currentState).toBe("HALF_OPEN");
    expect(cb.shouldBypass()).toBe(false);
    cb.recordSuccess();
    cb.recordSuccess();
    expect(cb.currentState).toBe("CLOSED");
  });

  it("releaseStuckProbe clears HALF_OPEN when probe never completed", () => {
    const cb = makeCircuit(1, 3);
    cb.recordFailure();
    cb.shouldProbe();
    expect(cb.currentState).toBe("HALF_OPEN");
    cb.releaseStuckProbe();
    expect(cb.currentState).toBe("OPEN");
    expect(cb.shouldBypass()).toBe(true);
  });

  it("fires onTransition once per state change", () => {
    const transitions: string[] = [];
    const cb = new CircuitBreaker(
      { openThreshold: 1, closeThreshold: 1 },
      (from, to) => transitions.push(`${from}->${to}`),
    );
    cb.recordFailure();
    expect(transitions).toEqual(["CLOSED->OPEN"]);
    cb.shouldProbe();
    expect(transitions).toEqual(["CLOSED->OPEN", "OPEN->HALF_OPEN"]);
    cb.recordSuccess();
    expect(transitions).toEqual([
      "CLOSED->OPEN",
      "OPEN->HALF_OPEN",
      "HALF_OPEN->CLOSED",
    ]);
  });
});
