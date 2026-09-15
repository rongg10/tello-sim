/* Live 3D view of the Tello simulator.
 *
 * Reads state from the Python process and sends ordinary SDK commands back.
 * It is a client like any other -- the same commands the tkinter GUI sends, and
 * the same ones the real drone would receive. Nothing here changes the physics.
 *
 * Coordinates. The simulator uses the Tello convention (x forward, y left,
 * z up); three.js is y-up. One mapping is used everywhere:
 *
 *     three.x = world.x        three.y = world.z        three.z = -world.y
 *
 * which stays right-handed, so orientations transfer without sign surprises.
 */

'use strict';

const W2T = (v) => new THREE.Vector3(v[0], v[2], -v[1]);

const state = {
  world: null,
  drones: [],
  selected: 0,
  camMode: 'orbit',
  droneScale: 2,
  last: null,       // most recent server payload
  prev: null,       // the one before, for interpolation
  lastAt: 0,
  connected: false,
  logSeq: 0,
};

// ---------------------------------------------------------------- renderer

const container = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputEncoding = THREE.sRGBEncoding;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
scene.fog = new THREE.Fog(0x0e1116, 14, 44);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.05, 200);
const fpvCamera = new THREE.PerspectiveCamera(72, innerWidth / innerHeight, 0.02, 200);

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  for (const cam of [camera, fpvCamera]) {
    cam.aspect = innerWidth / innerHeight;
    cam.updateProjectionMatrix();
  }
});

// ---------------------------------------------------------------- lighting

scene.add(new THREE.HemisphereLight(0x9fb4d8, 0x1a1f28, 0.75));

const keyLight = new THREE.DirectionalLight(0xffffff, 0.95);
keyLight.position.set(5, 9, 4);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 1;
keyLight.shadow.camera.far = 40;
Object.assign(keyLight.shadow.camera, { left: -10, right: 10, top: 10, bottom: -10 });
keyLight.shadow.bias = -0.0009;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x7fa4ff, 0.28);
fillLight.position.set(-6, 4, -5);
scene.add(fillLight);

// ---------------------------------------------------------------- orbiting
// Written by hand rather than pulled in as an addon: it is forty lines, and it
// keeps the page to a single vendored file that works with no network.

const orbit = { target: new THREE.Vector3(0, 0.8, 0), radius: 8, theta: 0.7, phi: 1.05 };

function applyOrbit() {
  const r = orbit.radius, p = Math.max(0.08, Math.min(Math.PI - 0.08, orbit.phi));
  camera.position.set(
    orbit.target.x + r * Math.sin(p) * Math.cos(orbit.theta),
    orbit.target.y + r * Math.cos(p),
    orbit.target.z + r * Math.sin(p) * Math.sin(orbit.theta),
  );
  camera.lookAt(orbit.target);
}

(function bindOrbit() {
  let dragging = false, panning = false, lastX = 0, lastY = 0;
  const el = renderer.domElement;

  el.addEventListener('pointerdown', (e) => {
    dragging = true;
    panning = e.shiftKey || e.button === 2;
    lastX = e.clientX; lastY = e.clientY;
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener('pointerup', (e) => {
    dragging = false;
    try { el.releasePointerCapture(e.pointerId); } catch (_) {}
  });
  el.addEventListener('contextmenu', (e) => e.preventDefault());
  el.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (panning) {
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
      const k = orbit.radius * 0.0016;
      orbit.target.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
    } else {
      orbit.theta -= dx * 0.006;
      orbit.phi -= dy * 0.006;
    }
  });
  el.addEventListener('wheel', (e) => {
    e.preventDefault();
    orbit.radius = Math.max(0.8, Math.min(40, orbit.radius * (1 + Math.sign(e.deltaY) * 0.11)));
  }, { passive: false });
})();

// ---------------------------------------------------------------- the drone
// Built from primitives to match the airframe on the desk: a Tello inside a
// spherical propeller cage. Real proportions -- the body is about 10 cm across
// and the cage about 18 -- with a display scale applied on top, because at true
// size in a 6 m room it is a speck.

const MAT = {
  body: new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: 0.55, metalness: 0.35 }),
  trim: new THREE.MeshStandardMaterial({ color: 0x121519, roughness: 0.7, metalness: 0.2 }),
  cage: new THREE.MeshStandardMaterial({ color: 0x1b1e24, roughness: 0.8, metalness: 0.1 }),
  prop: new THREE.MeshStandardMaterial({
    color: 0x39414d, roughness: 0.4, transparent: true, opacity: 0.85, side: THREE.DoubleSide,
  }),
  motor: new THREE.MeshStandardMaterial({ color: 0xb8892f, roughness: 0.35, metalness: 0.8 }),
};

function buildDrone(colour) {
  const g = new THREE.Group();
  const rotors = [];

  const body = new THREE.Mesh(new THREE.BoxGeometry(0.098, 0.041, 0.093), MAT.body);
  body.position.y = 0.004;
  body.castShadow = true;
  g.add(body);

  // A coloured strip along the top: the only way to tell drones apart at range.
  const stripe = new THREE.Mesh(
    new THREE.BoxGeometry(0.07, 0.005, 0.028),
    new THREE.MeshStandardMaterial({ color: colour, emissive: colour, emissiveIntensity: 0.45 }),
  );
  stripe.position.set(0.004, 0.026, 0);
  g.add(stripe);

  // Nose marker, so the heading is readable without reading the compass.
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.014, 0.03, 12), MAT.trim);
  nose.rotation.z = -Math.PI / 2;
  nose.position.set(0.062, 0.004, 0);
  g.add(nose);

  const ARM = 0.062;
  for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const x = sx * ARM, z = sz * ARM;

    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.058, 0.011, 0.016), MAT.trim);
    arm.position.set(x * 0.55, 0.002, z * 0.55);
    arm.rotation.y = -Math.atan2(z, x);
    g.add(arm);

    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.012, 0.024, 14), MAT.motor);
    motor.position.set(x, 0.012, z);
    motor.castShadow = true;
    g.add(motor);

    // Two thin blades. They are drawn as flat boxes because a real blade at
    // this size is a couple of pixels, and the point is to read the spin.
    const rotor = new THREE.Group();
    for (const angle of [0, Math.PI / 2]) {
      const blade = new THREE.Mesh(new THREE.BoxGeometry(0.076, 0.0016, 0.012), MAT.prop);
      blade.rotation.y = angle;
      rotor.add(blade);
    }
    rotor.position.set(x, 0.026, z);
    g.add(rotor);
    rotors.push(rotor);
  }

  // The cage: vertical hoops plus a couple of horizontal ones, as in the photo.
  const cage = new THREE.Group();
  const R = 0.128;
  for (let i = 0; i < 6; i++) {
    const hoop = new THREE.Mesh(new THREE.TorusGeometry(R, 0.0022, 5, 44), MAT.cage);
    hoop.rotation.y = (i / 6) * Math.PI;
    hoop.rotation.x = Math.PI / 2;
    cage.add(hoop);
  }
  for (const [y, scale] of [[0.055, 0.90], [-0.02, 0.99], [-0.085, 0.74]]) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(R * scale, 0.0022, 5, 48), MAT.cage);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = y;
    cage.add(ring);
  }
  cage.scale.y = 0.86;
  g.add(cage);

  g.userData.rotors = rotors;
  return g;
}

// A soft blob under the drone. Cheap, and it does more for reading altitude
// than the real shadow map does.
function buildGroundSpot(colour) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(0,0,0,0.5)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 128, 128);

  const spot = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({
      map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false,
    }),
  );
  spot.rotation.x = -Math.PI / 2;
  return spot;
}

const DRONE_COLOURS = [0x4da3ff, 0xff8a3d, 0x35d39a, 0xe06bd0, 0xf2d047, 0x7fd4ff];

// ---------------------------------------------------------------- the room

function makeLabel(text, colour, position) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 38px ui-monospace, Menlo, monospace';
  ctx.fillStyle = `#${colour.toString(16).padStart(6, '0')}`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 128, 34);

  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(canvas), transparent: true, depthTest: false,
  }));
  sprite.scale.set(0.62, 0.155, 1);
  sprite.position.copy(position);
  return sprite;
}

function buildWorld(payload) {
  const world = payload.world;
  const lo = world.bounds_lower, hi = world.bounds_upper;
  const group = new THREE.Group();

  const sizeX = hi[0] - lo[0], sizeY = hi[1] - lo[1], height = hi[2] - lo[2];
  const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;

  // Floor.
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(sizeX, sizeY),
    new THREE.MeshStandardMaterial({ color: 0x1a1f27, roughness: 0.95 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(cx, lo[2] + 0.001, -cy);
  floor.receiveShadow = true;
  group.add(floor);

  const grid = new THREE.GridHelper(Math.max(sizeX, sizeY), Math.round(Math.max(sizeX, sizeY)), 0x2f3a4a, 0x212932);
  grid.position.set(cx, lo[2] + 0.003, -cy);
  group.add(grid);

  // Four wall panels rather than a closed box: a box would put a translucent
  // face between the camera and the floor and wash the whole room out.
  const wallMat = new THREE.MeshBasicMaterial({
    color: 0x38455a, transparent: true, opacity: 0.10,
    side: THREE.DoubleSide, depthWrite: false,
  });
  const midY = lo[2] + height / 2;
  const walls = [
    { w: sizeX, x: cx, z: -hi[1], ry: 0 },
    { w: sizeX, x: cx, z: -lo[1], ry: 0 },
    { w: sizeY, x: lo[0], z: -cy, ry: Math.PI / 2 },
    { w: sizeY, x: hi[0], z: -cy, ry: Math.PI / 2 },
  ];
  for (const wall of walls) {
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(wall.w, height), wallMat);
    panel.position.set(wall.x, midY, wall.z);
    panel.rotation.y = wall.ry;
    group.add(panel);
  }

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(sizeX, height, sizeY)),
    new THREE.LineBasicMaterial({ color: 0x46556c }),
  );
  edges.position.set(cx, midY, -cy);
  group.add(edges);

  // An axis marker in one corner. The Tello convention (x forward, y left) is
  // easy to lose track of once the camera has been orbited, and every command
  // the drone accepts is expressed in it.
  const axes = new THREE.Group();
  const axisSpecs = [
    { dir: [1, 0, 0], colour: 0xe05252, label: '+x fwd' },
    { dir: [0, 1, 0], colour: 0x38c172, label: '+y left' },
    { dir: [0, 0, 1], colour: 0x4da3ff, label: '+z up' },
  ];
  for (const spec of axisSpecs) {
    const end = W2T(spec.dir).multiplyScalar(0.6);
    axes.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), end]),
      new THREE.LineBasicMaterial({ color: spec.colour }),
    ));
    axes.add(makeLabel(spec.label, spec.colour, end));
  }
  axes.position.set(lo[0] + 0.35, lo[2] + 0.02, -(lo[1] + 0.35));
  group.add(axes);

  // Obstacles.
  const obstacleMat = new THREE.MeshStandardMaterial({ color: 0x39424f, roughness: 0.85 });
  for (const o of world.obstacles || []) {
    let mesh;
    if (o.type === 'box') {
      const [x0, y0, z0] = o.lower, [x1, y1, z1] = o.upper;
      mesh = new THREE.Mesh(
        new THREE.BoxGeometry(x1 - x0, z1 - z0, y1 - y0), obstacleMat,
      );
      mesh.position.set((x0 + x1) / 2, (z0 + z1) / 2, -(y0 + y1) / 2);
    } else if (o.type === 'cylinder') {
      const h = o.z_max - o.z_min;
      mesh = new THREE.Mesh(new THREE.CylinderGeometry(o.radius, o.radius, h, 24), obstacleMat);
      mesh.position.set(o.center_xy[0], o.z_min + h / 2, -o.center_xy[1]);
    }
    if (!mesh) continue;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);

    // Outline the obstacle so its extent is readable against a dark floor.
    // Note: position and rotation on an Object3D are read-only bindings, so
    // they must be copied into, never assigned over.
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: 0x5a6a80 }),
    );
    outline.position.copy(mesh.position);
    outline.rotation.copy(mesh.rotation);
    group.add(outline);
  }

  // Mission pads, with their numbers painted on.
  for (const pad of world.mission_pads || []) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 256;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#101318'; ctx.fillRect(0, 0, 256, 256);
    ctx.strokeStyle = '#d9a441'; ctx.lineWidth = 10;
    ctx.strokeRect(12, 12, 232, 232);
    ctx.fillStyle = '#d9a441';
    ctx.font = 'bold 150px system-ui, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(pad.pad_id), 128, 136);

    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(pad.size, pad.size),
      new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(canvas) }),
    );
    mesh.rotation.x = -Math.PI / 2;
    mesh.rotation.z = -(pad.yaw || 0);
    mesh.position.set(pad.center[0], lo[2] + 0.006, -pad.center[1]);
    group.add(mesh);
  }

  // Frame the whole room from outside and above. A tighter, lower start looks
  // better in an empty box and puts the camera inside the furniture the moment
  // the room has interior walls.
  orbit.target.set(cx, lo[2] + 0.75, -cy);
  orbit.radius = Math.max(sizeX, sizeY) * 0.95 + 2.0;
  orbit.phi = 1.02;
  return group;
}

// ---------------------------------------------------------------- wind field
// Drawn because a draught that covers only part of the room is invisible in a
// single number, and that unevenness is exactly what breaks a formation.

const windGroup = new THREE.Group();
scene.add(windGroup);
const windArrows = [];

function ensureWindArrows(n) {
  const shaftMat = new THREE.MeshBasicMaterial({ color: 0x5f7fa8, transparent: true, opacity: 0.5 });
  while (windArrows.length < n) {
    const arrow = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 1, 5), shaftMat);
    shaft.position.y = 0.5;
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.022, 0.07, 8), shaftMat);
    head.position.y = 1;
    arrow.add(shaft, head);
    arrow.userData = { shaft, head };
    windGroup.add(arrow);
    windArrows.push(arrow);
  }
}

const UP = new THREE.Vector3(0, 1, 0);

function updateWind(samples) {
  ensureWindArrows(samples.length);
  windArrows.forEach((a, i) => (a.visible = i < samples.length));

  samples.forEach((s, i) => {
    const arrow = windArrows[i];
    const dir = W2T(s.v);
    const speed = dir.length();
    if (speed < 1e-4) { arrow.visible = false; return; }

    arrow.position.copy(W2T(s.p));
    arrow.quaternion.setFromUnitVectors(UP, dir.clone().normalize());

    const len = Math.min(0.14 + speed * 0.16, 0.6);
    arrow.userData.shaft.scale.y = len;
    arrow.userData.shaft.position.y = len / 2;
    arrow.userData.head.position.y = len;

    // Cool and faint in still air, warm and solid in a gust: the eye should
    // find the windy part of the room without reading any numbers.
    const t = Math.min(speed / 2.5, 1);
    arrow.userData.shaft.material.color.setRGB(
      0.26 + 0.66 * t, 0.40 + 0.26 * t, 0.58 - 0.35 * t,
    );
    arrow.userData.shaft.material.opacity = 0.20 + 0.55 * t;
  });
}

// ---------------------------------------------------------------- drone view

const views = [];   // one entry per drone: meshes, trail, interpolation targets

function makeView(index, name) {
  const colour = DRONE_COLOURS[index % DRONE_COLOURS.length];
  const model = buildDrone(colour);
  model.scale.setScalar(state.droneScale);
  scene.add(model);

  const spot = buildGroundSpot(colour);
  scene.add(spot);

  const trailGeom = new THREE.BufferGeometry();
  trailGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * 4000), 3));
  trailGeom.setDrawRange(0, 0);
  const trail = new THREE.Line(
    trailGeom,
    new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.72 }),
  );
  trail.frustumCulled = false;
  scene.add(trail);

  return {
    name, colour, model, spot, trail,
    trailCount: 0,
    position: new THREE.Vector3(),
    quaternion: new THREE.Quaternion(),
    targetPosition: new THREE.Vector3(),
    targetQuaternion: new THREE.Quaternion(),
    started: false,
    spin: 0,
  };
}

const BASIS = new THREE.Matrix4();

function applyDroneState(view, d) {
  view.targetPosition.copy(W2T(d.p));

  // Rebuild the orientation from the body axes the server sends. The model is
  // drawn facing +X with +Y up, so its local axes map to forward / up / right,
  // and right is the negation of the server's left.
  const forward = W2T(d.forward), up = W2T(d.up), right = W2T(d.left).negate();
  BASIS.makeBasis(forward, up, right);
  view.targetQuaternion.setFromRotationMatrix(BASIS);

  if (!view.started) {
    view.position.copy(view.targetPosition);
    view.quaternion.copy(view.targetQuaternion);
    view.started = true;
  }
  view.spinRate = d.motors ? 26 + d.rotor * 26 : 0;
  view.airborne = d.airborne;
}

function pushTrail(view) {
  const attr = view.trail.geometry.attributes.position;
  const max = attr.count;
  if (view.trailCount >= max) {
    // Full: drop the oldest half rather than the whole trail, so a long flight
    // does not blink back to nothing.
    const keep = Math.floor(max / 2);
    attr.array.copyWithin(0, (max - keep) * 3, max * 3);
    view.trailCount = keep;
  }
  const i = view.trailCount * 3;
  attr.array[i] = view.position.x;
  attr.array[i + 1] = view.position.y;
  attr.array[i + 2] = view.position.z;
  view.trailCount += 1;
  attr.needsUpdate = true;
  view.trail.geometry.setDrawRange(0, view.trailCount);
}

// ---------------------------------------------------------------- talking to the sim

async function postCommandTo(droneName, command) {
  try {
    await fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ drone: droneName, command }),
    });
  } catch (_) { /* the offline overlay already says so */ }
}

async function postCommand(command) {
  if (state.selected === 'all') {
    // Same order at the same instant, which is the whole point of a formation
    // test: any change in separation afterwards came from the air, not the plan.
    await Promise.all(state.drones.map((d) => postCommandTo(d.name, command)));
    return;
  }
  const drone = state.drones[state.selected];
  if (drone) await postCommandTo(drone.name, command);
}

/** Which drone the readouts describe. With "all" selected, show the first. */
function focusIndex() {
  return state.selected === 'all' ? 0 : state.selected;
}

async function loadWorld() {
  const payload = await (await fetch('/api/world')).json();
  state.world = payload;
  scene.add(buildWorld(payload));

  payload.drones.forEach((d, i) => views.push(makeView(i, d.name)));
  state.drones = payload.drones;

  const select = document.getElementById('droneSel');
  if (payload.drones.length > 1) {
    select.style.display = '';
    select.innerHTML =
      `<option value="all">all ${payload.drones.length} drones</option>` +
      payload.drones.map((d, i) => `<option value="${i}">${d.name}</option>`).join('');
    select.onchange = () => {
      state.selected = select.value === 'all' ? 'all' : +select.value;
    };
    state.selected = 'all';
    document.getElementById('sepRow').style.display = '';
  }

  // Every client has to put the drone into SDK mode before it will accept
  // anything, this page included. Skipping it is why the first build got
  // "error Not in SDK mode" back from every button.
  for (const d of payload.drones) {
    await postCommandTo(d.name, 'command');
    await postCommandTo(d.name, `speed ${ui.spd.value}`);
  }

  const w = payload.world;
  document.getElementById('banner').innerHTML =
    `<b>${w.name}</b> &nbsp;·&nbsp; ${w.bounds_upper[0] - w.bounds_lower[0]}` +
    ` × ${w.bounds_upper[1] - w.bounds_lower[1]} × ${w.bounds_upper[2] - w.bounds_lower[2]} m` +
    ` &nbsp;·&nbsp; ${describeWind(w.wind)}`;
}

function describeWind(wind) {
  if (!wind || wind.type === 'none') return 'still air';
  if (wind.type === 'constant') {
    const v = wind.velocity;
    return `steady wind ${Math.hypot(v[0], v[1], v[2]).toFixed(1)} m/s`;
  }
  if (wind.type === 'turbulent') return `turbulence σ ${(wind.sigma || 0).toFixed(1)} m/s`;
  if (wind.type === 'gust_burst') {
    const v = wind.velocity;
    return `gust ${Math.hypot(v[0], v[1], v[2]).toFixed(1)} m/s at t=${wind.start}s`;
  }
  if (wind.type === 'boundary_layer') return `boundary layer ${wind.speed_at_reference} m/s`;
  if (wind.type === 'wind_tunnel') {
    const v = wind.velocity;
    return `draught ${Math.hypot(v[0], v[1], v[2]).toFixed(1)} m/s over part of the room`;
  }
  if (wind.type === 'sum') return wind.fields.map(describeWind).join(' + ');
  return wind.type;
}

async function poll() {
  for (;;) {
    try {
      const payload = await (await fetch('/api/state')).json();
      state.prev = state.last;
      state.last = payload;
      state.lastAt = performance.now();
      state.connected = true;
      document.getElementById('offline').classList.remove('show');

      payload.drones.forEach((d, i) => { if (views[i]) applyDroneState(views[i], d); });
      updateWind(payload.wind || []);
      updateHud(payload);
      updateLog(payload.log || []);
    } catch (_) {
      state.connected = false;
      document.getElementById('offline').classList.add('show');
      await new Promise((r) => setTimeout(r, 500));
    }
    await new Promise((r) => setTimeout(r, 33));
  }
}

// ---------------------------------------------------------------- readouts

function updateHud(payload) {
  const d = payload.drones[focusIndex()];
  if (!d) return;
  const set = (id, text) => { document.getElementById(id).textContent = text; };

  set('droneName', state.selected === 'all' ? `${d.name} (of all)` : d.name);
  set('mAction', d.action);
  set('mBatt', `${d.battery.toFixed(1)} %`);
  set('mHeight', `${d.height_cm} cm`);
  set('mTof', `${d.tof_cm} cm`);
  set('mSpeed', `${Math.hypot(d.v[0], d.v[1], d.v[2]).toFixed(2)} m/s`);
  set('mYaw', `${d.yaw_deg.toFixed(0)}°`);
  set('mPos', `${d.p[0].toFixed(2)} ${d.p[1].toFixed(2)} ${d.p[2].toFixed(2)}`);
  set('mWind', `${Math.hypot(d.wind[0], d.wind[1], d.wind[2]).toFixed(2)} m/s`);
  set('mPad', d.pad < 0 ? 'none' : `m${d.pad}`);
  if (payload.min_separation !== null && payload.min_separation !== undefined) {
    set('mSep', `${payload.min_separation.toFixed(2)} m`);
  }

  const bar = document.getElementById('battBar');
  bar.style.width = `${Math.max(0, Math.min(100, d.battery))}%`;
  bar.style.background = d.battery > 40 ? '#38c172' : d.battery > 15 ? '#e0a33e' : '#e05252';

  document.getElementById('statusDot').classList.toggle('flying', d.airborne);
}

function updateLog(entries) {
  if (!entries.length) return;
  const newest = entries[entries.length - 1];
  if (newest.seq === state.logSeq) return;
  state.logSeq = newest.seq;

  document.getElementById('logLines').innerHTML = entries.slice().reverse().map((e) => {
    const failed = e.response && e.response.startsWith('error');
    const took = e.took === null ? '' : ` <span class="took">${e.took.toFixed(2)}s</span>`;
    const who = state.drones.length > 1 ? `${e.drone} ` : '';
    return `<div><span class="t">${e.t.toFixed(1)}s</span> ${who}` +
      `<span class="c">${e.command}</span> → ` +
      `<span class="${failed ? 'err' : 'ok'}">${e.response}</span>${took}</div>`;
  }).join('');
}

// ---------------------------------------------------------------- controls

const ui = {
  dist: document.getElementById('dist'),
  rot: document.getElementById('rot'),
  spd: document.getElementById('spd'),
  scale: document.getElementById('scale'),
};

function bindSlider(input, labelId, onChange, fireOnInit = true) {
  const label = document.getElementById(labelId);
  const apply = (fromUser) => {
    label.textContent = input.id === 'scale' ? (+input.value).toFixed(1) : input.value;
    if (onChange && (fromUser || fireOnInit)) onChange(+input.value);
  };
  input.addEventListener('input', () => apply(true));
  apply(false);
}

bindSlider(ui.dist, 'distVal');
bindSlider(ui.rot, 'rotVal');
// Not on init: the drone is not in SDK mode until loadWorld has said hello.
bindSlider(ui.spd, 'spdVal', (v) => postCommand(`speed ${v}`), false);
bindSlider(ui.scale, 'scaleVal', (v) => {
  state.droneScale = v;
  views.forEach((view) => view.model.scale.setScalar(v));
});

function moveCommand(kind) {
  if (kind === 'cw' || kind === 'ccw') return `${kind} ${ui.rot.value}`;
  return `${kind} ${ui.dist.value}`;
}

document.querySelectorAll('[data-cmd]').forEach((b) =>
  b.addEventListener('click', () => postCommand(b.dataset.cmd)));
document.querySelectorAll('[data-move]').forEach((b) =>
  b.addEventListener('click', () => postCommand(moveCommand(b.dataset.move))));

document.querySelectorAll('.camBtn').forEach((b) =>
  b.addEventListener('click', () => {
    state.camMode = b.dataset.cam;
    document.querySelectorAll('.camBtn').forEach((o) => o.classList.toggle('primary', o === b));
  }));

const KEYS = {
  w: () => moveCommand('forward'), s: () => moveCommand('back'),
  a: () => moveCommand('left'), d: () => moveCommand('right'),
  r: () => moveCommand('up'), f: () => moveCommand('down'),
  q: () => moveCommand('ccw'), e: () => moveCommand('cw'),
  t: () => 'takeoff', l: () => 'land', ' ': () => 'stop',
};

addEventListener('keydown', (event) => {
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;
  const make = KEYS[event.key.toLowerCase()];
  if (!make) return;
  event.preventDefault();
  postCommand(make());
});

// ---------------------------------------------------------------- main loop

const clock = new THREE.Clock();
let trailTimer = 0;

function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(clock.getDelta(), 0.1);

  for (const view of views) {
    if (!view.started) continue;

    // The server is polled at about 30 Hz and this draws at 60+, so ease
    // towards the last reported pose instead of stepping to it.
    const k = 1 - Math.pow(0.0008, dt);
    view.position.lerp(view.targetPosition, k);
    view.quaternion.slerp(view.targetQuaternion, k);

    view.model.position.copy(view.position);
    view.model.quaternion.copy(view.quaternion);

    for (const rotor of view.model.userData.rotors) {
      rotor.rotation.y += (view.spinRate || 0) * dt;
    }

    const ground = state.world ? state.world.world.bounds_lower[2] : 0;
    const altitude = Math.max(view.position.y - ground, 0);
    view.spot.position.set(view.position.x, ground + 0.004, view.position.z);
    const spread = 0.34 + altitude * 0.42;
    view.spot.scale.set(spread, spread, 1);
    view.spot.material.opacity = Math.max(0.06, 0.5 - altitude * 0.16);
  }

  trailTimer += dt;
  if (trailTimer > 0.05) {
    trailTimer = 0;
    for (const view of views) {
      if (view.started && view.airborne) pushTrail(view);
    }
  }

  const followed = views[focusIndex()];
  let active = camera;

  if (followed && followed.started && state.camMode === 'follow') {
    orbit.target.lerp(followed.position, 1 - Math.pow(0.02, dt));
  } else if (followed && followed.started && state.camMode === 'fpv') {
    // Sit just ahead of and above the body, looking where the nose points.
    const offset = new THREE.Vector3(0.16, 0.05, 0).applyQuaternion(followed.quaternion);
    fpvCamera.position.copy(followed.position).add(offset);
    fpvCamera.quaternion.copy(followed.quaternion);
    // The model faces +X; a three.js camera looks down -Z, so turn it to match.
    fpvCamera.rotateY(-Math.PI / 2);
    active = fpvCamera;
  }

  applyOrbit();
  renderer.render(scene, active);
}

loadWorld().then(() => { poll(); frame(); });
