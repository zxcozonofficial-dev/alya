/* Общее для всех сцен Али: подгонка под окно, бумага с каракулями, субтитры, чат, очередь заказов
   и поток событий /events от main.py chat. Страница сама решает, какие блоки у неё есть: функции
   ничего не делают, если нужного элемента на странице нет. */
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
if (params.has("light")) document.documentElement.classList.add("light");

// сцена нарисована под 1920×1080 (так её видит OBS); в окне другого размера — вписываем целиком
function fit() { document.documentElement.style.zoom = Math.min(innerWidth / 1920, innerHeight / 1080); }
addEventListener("resize", fit);
fit();

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------- каракули: то, что Аля рисует дымом в перерывах ---------- */
// фигура — массив штрихов, штрих — массив точек [x, y] вокруг (0, 0) в масштабе 1
const SHAPES = {
  heart: () => [Array.from({ length: 60 }, (_, i) => {
    const t = i / 59 * Math.PI * 2;
    return [Math.sin(t) ** 3, -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) / 16];
  })],
  star: () => [Array.from({ length: 11 }, (_, i) => {
    const a = -Math.PI / 2 + i * Math.PI / 5, r = i % 2 ? .45 : 1; return [r * Math.cos(a), r * Math.sin(a)];
  })],
  spiral: () => [Array.from({ length: 80 }, (_, i) => {
    const t = i / 79 * Math.PI * 5; return [t / 16 * Math.cos(t), t / 16 * Math.sin(t)];
  })],
  cat: () => [
    Array.from({ length: 41 }, (_, i) => { const t = i / 40 * Math.PI * 2; return [Math.cos(t), .82 * Math.sin(t)]; }),
    [[-.8, -.45], [-.62, -1.2], [-.2, -.78]], [[.8, -.45], [.62, -1.2], [.2, -.78]],
    [[-.35, -.05], [-.3, .05]], [[.35, -.05], [.3, .05]], [[-.15, .3], [0, .4], [.15, .3]],
  ],
};
function strokeShape(c, name, cx, cy, s, upto = 1, rng = Math.random) {
  // upto — доля нарисованного (для анимации «рисует на глазах»); rng — дрожание линии от руки
  const wob = (p) => p + (rng() - .5) * 1.2;
  const strokes = SHAPES[name]();
  const total = strokes.reduce((n, st) => n + st.length, 0);
  let left = Math.round(total * upto);
  for (const st of strokes) {
    if (left <= 1) break;
    const pts = st.slice(0, left);
    left -= st.length;
    c.beginPath();
    pts.forEach(([x, y], i) => (i ? c.lineTo(wob(cx + x * s), wob(cy + y * s)) : c.moveTo(cx + x * s, cy + y * s)));
    c.stroke();
  }
}
function paper(c, { margin = 1468, doodles = [], hole = null } = {}) {
  c.fillStyle = css("--paper"); c.fillRect(0, 0, 1920, 1080);
  if (margin) { c.strokeStyle = css("--margin"); c.lineWidth = 2; c.beginPath(); c.moveTo(margin, 0); c.lineTo(margin, 1080); c.stroke(); }
  c.strokeStyle = css("--doodle"); c.lineWidth = 2.2; c.lineCap = c.lineJoin = "round";
  for (const [name, x, y, s] of doodles) strokeShape(c, name, x, y, s);
  if (hole) c.clearRect(...hole);
}

/* ---------- субтитры ---------- */
let ended = true, quietTimer = null;
function say(text) {
  const box = $("say");
  if (!box) return;
  if (ended) { box.textContent = ""; ended = false; }
  clearTimeout(quietTimer);
  $("caption").classList.remove("quiet");
  text.trim().split(/\s+/).forEach((w, i) => {
    const s = document.createElement("span");
    s.className = "w"; s.textContent = w + " "; s.style.animationDelay = (i * 0.07) + "s";
    box.appendChild(s);
  });
  while (box.scrollHeight > box.clientHeight + 4 && box.children.length > 1) box.removeChild(box.firstChild);
}
function sayEnd() { ended = true; quietTimer = setTimeout(() => $("caption")?.classList.add("quiet"), 7000); }
function thinking() {
  if (!$("say")) return;
  ended = true; $("caption").classList.remove("quiet");
  $("say").innerHTML = '<span class="dots"><span>.</span><span>.</span><span>.</span></span>';
}

/* ---------- чат и заказы ---------- */
// цвета ников Twitch бывают слишком тёмными или кислотными для бумаги — подмешиваем цвет чернил
function inkify(color) {
  const ink = css("--nick-mix").split(",").map(Number);
  if (!/^#[0-9a-f]{6}$/i.test(color)) return `rgb(${ink})`;
  const c = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16));
  return "rgb(" + c.map((v, i) => Math.round(v * .62 + ink[i] * .38)).join(",") + ")";
}
function chatMsg(user, text, color) {
  const list = $("chat-list");
  if (!list) return;
  $("chat-empty")?.remove();
  const d = document.createElement("div"), b = document.createElement("b");
  d.className = "msg"; b.textContent = user; b.style.color = inkify(color);
  d.append(b, document.createTextNode(text));
  list.appendChild(d);
  while (list.children.length > 25) list.removeChild(list.firstChild);
}
function showQueue(items) {
  const el = $("queue");
  if (!el) return;
  el.textContent = "";
  if (!items.length) { el.append("request a map: !req <link>"); return; }
  const b = document.createElement("b");
  b.textContent = `${items.length} in queue`;
  el.append(b, ` · next: ${items[0].title} [${items[0].version}] from ${items[0].user}`);
}

/* ---------- голос: только на странице с ?voice=1 (так её открывает OBS) ---------- */
// OBS берёт звук браузерного источника отдельно от «Звука рабочего стола», поэтому Discord в стрим не попадает.
// Куски идут друг за другом без щелей: каждый ставится в очередь сразу после предыдущего
let audioCtx = null, audioAt = 0;
function playVoice(sr, pcm) {
  if (!params.has("voice")) return;
  audioCtx = audioCtx || new AudioContext();
  const raw = atob(pcm), n = raw.length >> 1, buf = audioCtx.createBuffer(1, n, sr), ch = buf.getChannelData(0);
  for (let i = 0; i < n; i++) {
    const v = raw.charCodeAt(2 * i) | (raw.charCodeAt(2 * i + 1) << 8);
    ch[i] = (v > 32767 ? v - 65536 : v) / 32768;
  }
  const src = audioCtx.createBufferSource();
  src.buffer = buf; src.connect(audioCtx.destination);
  audioAt = Math.max(audioAt, audioCtx.currentTime + 0.02);
  src.start(audioAt);
  audioAt += buf.duration;
}

function connectEvents() {
  const es = new EventSource("/events");
  es.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.type === "say") say(d.text);
    else if (d.type === "say_end") sayEnd();
    else if (d.type === "thinking") thinking();
    else if (d.type === "emotion" && $("emo")) $("emo").textContent = "(" + (d.emotion === "embarrassed" ? "flustered" : d.emotion) + ")";
    else if (d.type === "chat") chatMsg(d.user, d.text, d.color);
    else if (d.type === "requests") showQueue(d.items);
    else if (d.type === "channel" && $("chat-channel")) {
      $("chat-channel").textContent = "twitch.tv/" + d.name;
      if ($("chat-empty")) $("chat-empty").textContent = "quiet in here. say hi, she reads every message";
    }
    else if (d.type === "audio") playVoice(d.sr, d.pcm);
  };
  es.onerror = () => { es.close(); setTimeout(connectEvents, 2000); };
}
connectEvents();
