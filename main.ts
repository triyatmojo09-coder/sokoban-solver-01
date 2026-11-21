// Room-wise push-optimal Sokoban solver.
//
// Idea: the generator builds a level as one 1-wide hallway with 1-4 small rooms
// off it, each room holding 3-4 boxes and as many goals, with exactly one door.
// Boxes never leave their room, and a room finished never shuts the player in,
// so the level's push optimum is the sum of the rooms' optima and the rooms can
// be solved independently, one at a time, and the walks between them stitched
// together. Each room is solved push-optimally with A* over pushes:
//
//   state  = sorted box cells + the player's reachable region (canonicalised
//            to the region's lowest cell; the exact player cell inside a region
//            never changes which pushes are possible)
//   cost   = pushes; heuristic = minimum assignment of boxes to goals by
//            shortest walk distance (Hungarian over <= 4 boxes, admissible and
//            consistent, so the first goal state popped is push-optimal)
//   prune  = dead squares (a box that can never be pulled back to a goal),
//            frozen 2x2 blocks, and any state whose boxes cannot all reach a goal
//
// The full move string is replayed on the level before it is written out, so a
// bug can only ever cost the instance, never make the answer illegal.
//
// Fully deterministic: no clock, no environment, no randomness. Every search is
// cut off by a fixed expansion count, never by time, so two runs of the program
// produce byte-identical outputs.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// [dx, dy, walk letter, push letter]; order is fixed everywhere, so every
// search and every walk is reproducible.
const DIRS: { dx: number; dy: number; walk: string; push: string }[] = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
];

const MAX_EXPANSIONS_PER_ROOM = 300000;
const MAX_EXPANSIONS_GLOBAL = 300000;
const MAX_SOLUTION_BYTES = 65000;

// ---------------------------------------------------------------------------
// Board representation
// ---------------------------------------------------------------------------

interface Level {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[];
  goalSet: Uint8Array; // 1 = goal
  boxes: number[]; // sorted
  player: number;
}

function parseLevel(text: string): Level {
  const lines = text
    .split('\n')
    .map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l))
    .filter((l) => l.length > 0);
  const h = lines.length;
  const w = Math.max(...lines.map((l) => l.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const goalSet = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const line = lines[y];
    for (let x = 0; x < line.length; x++) {
      const c = line[x];
      const i = y * w + x;
      if (c === '#') continue;
      wall[i] = 0;
      if (c === '.' || c === '*' || c === '+') {
        goals.push(i);
        goalSet[i] = 1;
      }
      if (c === '$' || c === '*') boxes.push(i);
      if (c === '@' || c === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

// Neighbour table: nbr[i * 4 + d] is the cell reached from i in direction d,
// or -1 when that step leaves the board.
function neighbourTable(w: number, h: number): Int32Array {
  const nbr = new Int32Array(w * h * 4).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      for (let d = 0; d < 4; d++) {
        const nx = x + DIRS[d].dx;
        const ny = y + DIRS[d].dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) nbr[i * 4 + d] = ny * w + nx;
      }
    }
  }
  return nbr;
}

// Shortest walk distances from `from` over floor cells (ignoring boxes).
function bfsDistances(
  wall: Uint8Array,
  nbr: Int32Array,
  from: number,
): Int32Array {
  const dist = new Int32Array(wall.length).fill(-1);
  const queue = [from];
  dist[from] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi];
    for (let d = 0; d < 4; d++) {
      const n = nbr[c * 4 + d];
      if (n < 0 || wall[n] || dist[n] >= 0) continue;
      dist[n] = dist[c] + 1;
      queue.push(n);
    }
  }
  return dist;
}

// The cells the player can reach from `from` without crossing a wall or a box.
function computeRegion(
  wall: Uint8Array,
  nbr: Int32Array,
  boxSet: Set<number>,
  from: number,
): { mask: Uint8Array; min: number } {
  const mask = new Uint8Array(wall.length);
  const queue = [from];
  mask[from] = 1;
  let min = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi];
    if (c < min) min = c;
    for (let d = 0; d < 4; d++) {
      const n = nbr[c * 4 + d];
      if (n < 0 || wall[n] || boxSet.has(n) || mask[n]) continue;
      mask[n] = 1;
      queue.push(n);
    }
  }
  return { mask, min };
}

// Shortest walk from `from` to `to` (floor, avoiding boxes), as a string of
// lowercase move letters, or null when unreachable. Neighbours are tried in
// the fixed DIRS order, so the path is deterministic.
function walkPath(
  wall: Uint8Array,
  nbr: Int32Array,
  boxSet: Set<number>,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const prevDir = new Int8Array(wall.length).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi];
    for (let d = 0; d < 4; d++) {
      const n = nbr[c * 4 + d];
      if (n < 0 || wall[n] || boxSet.has(n) || prev[n] >= 0) continue;
      prev[n] = c;
      prevDir[n] = d;
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          s = DIRS[prevDir[cur]].walk + s;
          cur = prev[cur];
        }
        return s;
      }
      queue.push(n);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Push-optimal A* over one board (a room, or the whole level as a fallback)
// ---------------------------------------------------------------------------

interface BoardSpec {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall; everything outside the board's floor is a wall
  floor: number[];
  goals: number[];
  boxes: number[]; // sorted, initial
  start: number; // player start cell
  door: number; // door cell the player must be able to reach at the end, or -1
}

interface Solved {
  moves: string;
  boxes: number[]; // final, sorted
  player: number;
  pushes: number;
  expansions: number;
}

interface SearchNode {
  boxes: number[];
  player: number;
  regionMin: number;
  g: number;
  h: number;
  f: number;
  parent: SearchNode | null;
  trans: string;
  seq: number;
}

// Minimal binary heap ordered by (f, h, seq).
class Heap {
  private a: SearchNode[] = [];
  get size(): number {
    return this.a.length;
  }
  push(n: SearchNode): void {
    const a = this.a;
    a.push(n);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (cmp(a[i], a[p]) < 0) {
        const t = a[i];
        a[i] = a[p];
        a[p] = t;
        i = p;
      } else break;
    }
  }
  pop(): SearchNode | undefined {
    const a = this.a;
    if (a.length === 0) return undefined;
    const top = a[0];
    const last = a.pop() as SearchNode;
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < a.length && cmp(a[l], a[m]) < 0) m = l;
        if (r < a.length && cmp(a[r], a[m]) < 0) m = r;
        if (m === i) break;
        const t = a[i];
        a[i] = a[m];
        a[m] = t;
        i = m;
      }
    }
    return top;
  }
}

function cmp(x: SearchNode, y: SearchNode): number {
  if (x.f !== y.f) return x.f - y.f;
  if (x.h !== y.h) return x.h - y.h;
  return x.seq - y.seq;
}

function allPermutations(n: number): number[][] {
  const out: number[][] = [];
  const a: number[] = [];
  for (let i = 0; i < n; i++) a.push(i);
  const gen = (k: number): void => {
    if (k === 1) {
      out.push(a.slice());
      return;
    }
    for (let i = 0; i < k; i++) {
      gen(k - 1);
      const j = k % 2 === 1 ? 0 : i;
      const t = a[j];
      a[j] = a[k - 1];
      a[k - 1] = t;
    }
  };
  gen(n);
  return out;
}

function solveBoard(
  spec: BoardSpec,
  requireExit: boolean,
  maxExpansions: number,
): Solved | null {
  const { w, h, wall, floor, goals, boxes: startBoxes, start, door } = spec;
  const n = goals.length;
  if (startBoxes.length !== n) return null;
  const nbr = neighbourTable(w, h);
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;

  // Walk distances from every goal (boxes ignored).
  const distFromGoal: Int32Array[] = goals.map((g) =>
    bfsDistances(wall, nbr, g),
  );

  // Dead squares: start from the goals and grow by reverse pulls. A square s
  // joins the live set when a box on it could be pushed onto a live square t
  // while the player can reach the stance s - (t - s) from the start, ignoring
  // every box but the one on s. Reachability ignoring boxes over-approximates
  // the real game, so a square never marked live is truly dead.
  const live = new Uint8Array(w * h);
  for (const g of goals) live[g] = 1;
  const reachCache = new Map<number, Uint8Array>();
  const reachableFromStart = (blocked: number): Uint8Array => {
    const cached = reachCache.get(blocked);
    if (cached) return cached;
    const mask = new Uint8Array(w * h);
    const queue = [start];
    mask[start] = 1;
    for (let qi = 0; qi < queue.length; qi++) {
      const c = queue[qi];
      for (let d = 0; d < 4; d++) {
        const nn = nbr[c * 4 + d];
        if (nn < 0 || wall[nn] || nn === blocked || mask[nn]) continue;
        mask[nn] = 1;
        queue.push(nn);
      }
    }
    reachCache.set(blocked, mask);
    return mask;
  };
  let changed = true;
  while (changed) {
    changed = false;
    for (const s of floor) {
      if (live[s]) continue;
      for (let d = 0; d < 4; d++) {
        const t = nbr[s * 4 + d];
        if (t < 0 || wall[t] || !live[t]) continue;
        const stance = nbr[s * 4 + (d ^ 1)];
        if (stance < 0 || wall[stance]) continue;
        if (reachableFromStart(s)[stance]) {
          live[s] = 1;
          changed = true;
          break;
        }
      }
    }
  }
  for (const b of startBoxes) {
    if (!live[b]) return null; // a box starts on a dead square
  }

  // 2x2 blocks, indexed by the cells they contain.
  const blocksByCell = new Map<number, number[][]>();
  for (let y = 0; y < h - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const cells = [y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1];
      for (const c of cells) {
        const list = blocksByCell.get(c);
        if (list) list.push(cells);
        else blocksByCell.set(c, [cells]);
      }
    }
  }
  const frozen2x2 = (boxSet: Set<number>, moved: number): boolean => {
    const lists = blocksByCell.get(moved);
    if (!lists) return false;
    for (const cells of lists) {
      let allBlocked = true;
      let badBox = false;
      for (const c of cells) {
        if (wall[c]) continue;
        if (boxSet.has(c)) {
          if (!goalSet[c]) badBox = true;
          continue;
        }
        allBlocked = false;
        break;
      }
      if (allBlocked && badBox) return true;
    }
    return false;
  };

  // Heuristic: minimum assignment of boxes to goals by walk distance.
  // Exact Hungarian by permutation for small n; for larger boards the sum of
  // each box's nearest-goal distance (still admissible).
  const perms = n <= 5 ? allPermutations(n) : null;
  const heuristic = (boxes: number[]): number => {
    if (perms) {
      let best = Infinity;
      for (const p of perms) {
        let s = 0;
        for (let j = 0; j < n; j++) {
          const d = distFromGoal[p[j]][boxes[j]];
          if (d < 0) {
            s = Infinity;
            break;
          }
          s += d;
        }
        if (s < best) best = s;
      }
      return best;
    }
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (let g = 0; g < n; g++) {
        const d = distFromGoal[g][b];
        if (d >= 0 && d < m) m = d;
      }
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  };

  const isGoalBoxes = (boxes: number[]): boolean => {
    for (const b of boxes) if (!goalSet[b]) return false;
    return true;
  };

  let seq = 0;
  const startSet = new Set<number>(startBoxes);
  const startRegion = computeRegion(wall, nbr, startSet, start);
  const h0 = heuristic(startBoxes);
  if (h0 === Infinity) return null;
  const heap = new Heap();
  heap.push({
    boxes: startBoxes.slice(),
    player: start,
    regionMin: startRegion.min,
    g: 0,
    h: h0,
    f: h0,
    parent: null,
    trans: '',
    seq: seq++,
  });
  const closed = new Map<string, number>();
  closed.set(startBoxes.join(',') + '|' + startRegion.min, 0);

  let expansions = 0;
  while (heap.size > 0) {
    const node = heap.pop() as SearchNode;
    if (expansions >= maxExpansions) return null;
    expansions++;
    if (isGoalBoxes(node.boxes)) {
      if (!requireExit || door < 0) return finish(node, expansions);
      // The player's region for this state: recompute from the node's player
      // cell (the exact cell is known: after a push the player stands where
      // the pushed box was).
      const region = computeRegion(wall, nbr, new Set<number>(node.boxes), node.player);
      if (region.mask[door]) return finish(node, expansions);
      // Otherwise this goal state cannot exit; keep searching.
    }
    const boxSet = new Set<number>(node.boxes);
    const region = computeRegion(wall, nbr, boxSet, node.player);
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const b = node.boxes[bi];
      for (let d = 0; d < 4; d++) {
        const dest = nbr[b * 4 + d];
        if (dest < 0 || wall[dest] || boxSet.has(dest)) continue;
        const stance = nbr[b * 4 + (d ^ 1)];
        if (stance < 0 || wall[stance] || boxSet.has(stance)) continue;
        if (!region.mask[stance]) continue;
        if (!live[dest]) continue; // dead square (dest is not a goal)
        const nb = node.boxes.slice();
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        const nbSet = new Set<number>(nb);
        if (frozen2x2(nbSet, dest)) continue;
        const h2 = heuristic(nb);
        if (h2 === Infinity) continue;
        const walk = walkPath(wall, nbr, boxSet, node.player, stance);
        if (walk === null) continue; // stance is in the region, so unreachable
        const g2 = node.g + 1;
        const reg2 = computeRegion(wall, nbr, nbSet, b);
        const key = nb.join(',') + '|' + reg2.min;
        const prev = closed.get(key);
        if (prev !== undefined && prev <= g2) continue;
        closed.set(key, g2);
        heap.push({
          boxes: nb,
          player: b,
          regionMin: reg2.min,
          g: g2,
          h: h2,
          f: g2 + h2,
          parent: node,
          trans: walk + DIRS[d].push,
          seq: seq++,
        });
      }
    }
  }
  return null;

  function finish(goal: SearchNode, exp: number): Solved {
    const parts: string[] = [];
    let cur: SearchNode | null = goal;
    while (cur !== null) {
      parts.push(cur.trans);
      cur = cur.parent;
    }
    parts.reverse();
    return {
      moves: parts.join(''),
      boxes: goal.boxes.slice(),
      player: goal.player,
      pushes: goal.g,
      expansions: exp,
    };
  }
}

// ---------------------------------------------------------------------------
// Level structure: hallway + rooms
// ---------------------------------------------------------------------------

interface Room {
  door: number; // door cell (in the wall row, part of the room)
  entry: number; // hallway cell outside the door; the player enters from here
  board: BoardSpec;
}

// Split the level into its rooms: connected floor components off the hallway
// row (the row the player starts on). Each component holds one door cell.
function detectRooms(level: Level): Room[] {
  const { w, h, wall, boxes, goals } = level;
  const hallRow = (level.player / w) | 0;
  const nbr = neighbourTable(w, h);
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  for (let i = 0; i < w * h; i++) {
    if (wall[i] || seen[i]) continue;
    if (((i / w) | 0) === hallRow) continue;
    const cells: number[] = [];
    const stack = [i];
    seen[i] = 1;
    while (stack.length > 0) {
      const c = stack.pop() as number;
      cells.push(c);
      for (let d = 0; d < 4; d++) {
        const n = nbr[c * 4 + d];
        if (n < 0 || wall[n] || seen[n]) continue;
        if (((n / w) | 0) === hallRow) continue;
        seen[n] = 1;
        stack.push(n);
      }
    }
    cells.sort((a, b) => a - b);
    // The door: the room cell next to a hallway floor cell.
    let door = -1;
    let entry = -1;
    for (const c of cells) {
      for (let d = 0; d < 4; d++) {
        const n = nbr[c * 4 + d];
        if (n < 0 || wall[n]) continue;
        if (((n / w) | 0) === hallRow) {
          door = c;
          entry = n;
          break;
        }
      }
      if (door >= 0) break;
    }
    if (door < 0) continue; // a component with no door is not a room
    const cellSet = new Set<number>(cells);
    const roomBoxes = boxes.filter((b) => cellSet.has(b));
    const roomGoals = goals.filter((g) => cellSet.has(g));
    if (roomBoxes.length === 0 || roomBoxes.length !== roomGoals.length) continue;
    const roomWall = new Uint8Array(w * h).fill(1);
    for (const c of cells) roomWall[c] = 0;
    roomWall[entry] = 0;
    rooms.push({
      door,
      entry,
      board: {
        w,
        h,
        wall: roomWall,
        floor: [...cells, entry],
        goals: roomGoals,
        boxes: roomBoxes,
        start: entry,
        door,
      },
    });
  }
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

// Replay a move string on the level; returns whether it is legal and solves
// the level, and how many pushes it took.
function replay(
  level: Level,
  moves: string,
): { legal: boolean; solved: boolean; pushes: number } {
  const nbr = neighbourTable(level.w, level.h);
  const boxSet = new Set<number>(level.boxes);
  let player = level.player;
  let pushes = 0;
  for (const ch of moves) {
    const lower = ch.toLowerCase();
    let d = -1;
    for (let i = 0; i < 4; i++) {
      if (DIRS[i].walk === lower) {
        d = i;
        break;
      }
    }
    if (d < 0) return { legal: false, solved: false, pushes };
    const next = nbr[player * 4 + d];
    if (next < 0 || level.wall[next]) return { legal: false, solved: false, pushes };
    if (boxSet.has(next)) {
      const dest = nbr[next * 4 + d];
      if (dest < 0 || level.wall[dest] || boxSet.has(dest)) {
        return { legal: false, solved: false, pushes };
      }
      boxSet.delete(next);
      boxSet.add(dest);
      pushes++;
    }
    player = next;
  }
  let solved = true;
  for (const b of boxSet) {
    if (!level.goalSet[b]) {
      solved = false;
      break;
    }
  }
  return { legal: true, solved, pushes };
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = detectRooms(level);
  if (rooms.length > 0) {
    let moves = '';
    let player = level.player;
    const curBoxes = level.boxes.slice();
    for (const room of rooms) {
      const boxSet = new Set<number>(curBoxes);
      const nbr = neighbourTable(level.w, level.h);
      const walk = walkPath(level.wall, nbr, boxSet, player, room.entry);
      if (walk === null) return '';
      moves += walk;
      player = room.entry;
      const solved =
        solveBoard(room.board, true, MAX_EXPANSIONS_PER_ROOM) ??
        solveBoard(room.board, false, MAX_EXPANSIONS_PER_ROOM);
      if (solved === null) return '';
      moves += solved.moves;
      const roomBoxSet = new Set<number>(room.board.boxes);
      const rest = curBoxes.filter((b) => !roomBoxSet.has(b));
      curBoxes.length = 0;
      curBoxes.push(...rest, ...solved.boxes);
      curBoxes.sort((a, b) => a - b);
      player = solved.player;
    }
    if (moves.length > MAX_SOLUTION_BYTES) return '';
    const check = replay(level, moves);
    if (check.legal && check.solved) return moves;
    return '';
  }
  // No rooms detected: fall back to solving the whole level as one board.
  const floor: number[] = [];
  for (let i = 0; i < level.w * level.h; i++) if (!level.wall[i]) floor.push(i);
  const solved = solveBoard(
    {
      w: level.w,
      h: level.h,
      wall: level.wall,
      floor,
      goals: level.goals,
      boxes: level.boxes,
      start: level.player,
      door: -1,
    },
    false,
    MAX_EXPANSIONS_GLOBAL,
  );
  if (solved === null || solved.moves.length > MAX_SOLUTION_BYTES) return '';
  const check = replay(level, solved.moves);
  if (check.legal && check.solved) return solved.moves;
  return '';
}

// ---------------------------------------------------------------------------
// Entry point: solve every instance under /input, write /output/<id>/solution.txt
// ---------------------------------------------------------------------------

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), moves + '\n');
}
