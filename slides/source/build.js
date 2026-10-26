// Six slides for Nan Fung's business team, written for readers who are not technical: what the London Office Monitor
// is, and how it helps with time saving, long-term data management and opportunity discovery.
// Usage: PPTX_SKILL=<pptx skill dir> node build.js <project dir> <out.pptx>
// Needs pptxgenjs, react, react-dom, react-icons and sharp on the module path (NODE_PATH or a local node_modules).
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const USAGE = "Usage: PPTX_SKILL=<pptx skill dir> node build.js <project dir> <out.pptx>";
const [PROJECT, OUT] = process.argv.slice(2);
if (!PROJECT || !OUT || !process.env.PPTX_SKILL) {
  console.error(USAGE);
  process.exit(2);
}
const { applyTheme } = require(path.join(process.env.PPTX_SKILL, "scripts", "apply_theme.js"));

const THEME = {
  name: "London Office Monitor",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2430", lt1: "FFFFFF", dk2: "23395B", lt2: "EEF1F5",
    accent1: "C8553D", accent2: "2E7D6B", accent3: "8A97A8", accent4: "E3B23C", accent5: "4A6FA5", accent6: "5B6472",
    hlink: "4A6FA5", folHlink: "5B6472",
  },
};
const HEX = THEME.colors;
const FOOTER = "London Office Monitor · proof of concept";

// The app's own screenshot, cropped to the app window (a pixel box in the 2880 x 1800 original) and scaled down to a
// width that stays sharp at the size it is placed (~400 dpi) while keeping the file small.
const APP_SHOT = { file: "docs/screenshots/chat-answer.png", crop: { left: 0, top: 0, width: 2780, height: 1730 } };
const SHOT_WIDTH_PX = 1600;

// The tour on the last slide: three screens of the web app, captured on 26 Oct 2026 and cropped (pixel boxes in the
// 2880 x 1800 originals) to the same 4:3 shape so the panels line up. Each matches a use on the first slide.
const TOUR = [
  { file: "docs/screenshots/chat-chart.png", crop: { left: 745, top: 205, width: 2025, height: 1519 },
    icon: fa.FaComments, head: "Chat", text: "Ask anything: every broker's figure, a chart and the sources",
    alt: "Chat: City Core prime rent by broker on a chart, with its sources" },
  { file: "docs/screenshots/weekly-brief.png", crop: { left: 862, top: 23, width: 1880, height: 1410 },
    icon: fa.FaNewspaper, head: "Weekly brief", text: "Read a draft of what changed and what it means for us",
    alt: "The weekly brief: what changed and the news themes, each linked to its source" },
  { file: "docs/screenshots/team-list.png", crop: { left: 749, top: 152, width: 1987, height: 1490 },
    icon: fa.FaBinoculars, head: "Opportunities", text: "Discover leads; approved ones are shared with the team",
    alt: "The Opportunities page: an approved opportunity card, marked as a demo example" },
];

// Times measured with the default models (GLM-5.3 answers, DeepSeek V4.1 Flash investigates), each up to the moment
// the result was ready. Answers: 16 runs in evals/results/20261024T114708Z, 20261024T115443Z, 20261024T115620Z and
// 20261025T143050Z, median 13 s. Brief: 31 s and 76 s in the same files. Scan: 56 s, 78 s and 104 s in
// 20261024T113722Z and 20261024T114708Z, plus 71 s and 115 s in the scan telemetry of 24 Oct 2026.
// Model cost per task across those runs: US$0.004 to US$0.07.
const TIMES = { answer: "~15 sec", brief: "~1 min", scan: "~2 min" };
const TIMES_NOTE = "Measured in test runs on 24–25 Oct 2026: answers median 13 s (16 runs), brief 31–76 s, scan 56–115 s " +
                   "until the cards are ready. How long the same work takes by hand is still to be measured.";

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size }));
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

async function screenshot({ file, crop }) {
  const png = await sharp(path.join(PROJECT, file)).extract(crop).resize({ width: SHOT_WIDTH_PX }).png().toBuffer();
  return { data: "image/png;base64," + png.toString("base64"), aspect: crop.width / crop.height };
}

// pptxgenjs converts option objects in place, so every shape gets a fresh shadow.
const softShadow = () => ({ type: "outer", blur: 6, offset: 2, angle: 90, color: "000000", opacity: 0.25 });

// Keeps a figure on the same line as its unit ("45,000 sq ft").
const NBSP = " ";

// Text that is not in a placeholder: a real text box, flush with the shapes it sits on.
const box = (opts) => ({ isTextBox: true, margin: 0, valign: "top", ...opts });

function defineLayouts(pres, C) {
  const kicker = (color) => ({ placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.4, w: 9, h: 0.28,
    fontSize: 12, bold: true, color, fontFace: THEME.bodyFontFace, valign: "middle", align: "left", margin: 0 },
    text: "" } });
  const title = (h, fontSize, color) => ({ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.7, w: 9,
    h, fontSize, bold: true, color, fontFace: THEME.headFontFace, valign: "top", align: "left", margin: 0 },
    text: "" } });
  const footer = (color) => ({ text: { text: FOOTER, options: { x: 0.5, y: 5.2, w: 6, h: 0.3, fontSize: 10, color,
                                                                 margin: 0, valign: "middle" } } });
  // Dark opener: room for a two-line title.
  pres.defineSlideMaster({
    title: "OPENER",
    background: { color: HEX.dk2 },
    objects: [kicker(C.accent4), title(1.05, 32, C.background1), footer(C.accent3)],
    slideNumber: { x: 9.1, y: 5.2, w: 0.4, h: 0.3, fontSize: 10, color: HEX.accent3 },
  });
  pres.defineSlideMaster({
    title: "DARK",
    background: { color: HEX.dk2 },
    objects: [kicker(C.accent4), title(0.6, 30, C.background1), footer(C.accent3)],
    slideNumber: { x: 9.1, y: 5.2, w: 0.4, h: 0.3, fontSize: 10, color: HEX.accent3 },
  });
  pres.defineSlideMaster({
    title: "LIGHT",
    background: { color: HEX.lt1 },
    objects: [kicker(C.accent1), title(0.6, 28, C.text2), footer(C.accent6)],
    slideNumber: { x: 9.1, y: 5.2, w: 0.4, h: 0.3, fontSize: 10, color: HEX.accent6 },
  });
}

function footnote(s, C, text, y = 4.85) {
  s.addText(text, box({ x: 0.5, y, w: 9, h: 0.32, fontSize: 10, color: C.accent6, objectName: "Footnote" }));
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
  pres.title = "London Office Monitor: what it does for the business team";
  pres.author = "Cody Chan";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  defineLayouts(pres, C);

  // ---------- 1. What it is (dark) ----------
  pres.addSection({ title: "What it is" });
  let s = pres.addSlide({ masterName: "OPENER", sectionTitle: "What it is" });
  s.addText("WHAT IT IS", { placeholder: "kicker", charSpacing: 1 });
  s.addText("An AI research assistant for the London office market", { placeholder: "title" });
  s.addText("It saves time, keeps a lasting record of the market, and flags opportunities that fit Nan Fung.",
            box({ x: 0.5, y: 1.9, w: 9, h: 0.35, fontSize: 16, color: C.background2, objectName: "Lead" }));
  const uses = [
    [fa.FaComments, "Ask", "Any market question, answered with every broker's figure and its source"],
    [fa.FaNewspaper, "Read", "A weekly brief: what changed, and what it means for our buildings"],
    [fa.FaBinoculars, "Discover", "Opportunities that fit our mandate. Only those the analyst approves reach the team"],
  ];
  for (let i = 0; i < uses.length; i++) {
    const y = 2.5 + i * 0.85;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y, w: 0.52, h: 0.52, fill: { color: C.accent4 }, line: { color: C.accent4 },
                                   objectName: uses[i][1] + " icon circle" });
    s.addImage({ data: await icon(uses[i][0], HEX.dk2), x: 0.63, y: y + 0.13, w: 0.26, h: 0.26,
                 altText: uses[i][1] + " icon" });
    s.addText(uses[i][1], box({ x: 1.2, y: y - 0.02, w: 3.9, h: 0.3, fontSize: 16, bold: true, color: C.background1 }));
    s.addText(uses[i][2], box({ x: 1.2, y: y + 0.28, w: 3.9, h: 0.5, fontSize: 14, color: C.background2 }));
  }
  const shot = await screenshot(APP_SHOT);
  const shotH = 2.35, shotW = shotH * shot.aspect, shotX = 9.5 - shotW, shotY = 2.48;
  s.addShape(pres.shapes.RECTANGLE, { x: shotX, y: shotY, w: shotW, h: shotH, fill: { color: C.background1 },
                                      line: { color: C.background1 }, shadow: softShadow(), objectName: "Screenshot frame" });
  s.addImage({ data: shot.data, x: shotX, y: shotY, w: shotW, h: shotH,
               altText: "The web app answering \"What's City prime rent now, and do the brokers agree?\" with each " +
                        "broker's figure and a link to its source" });
  s.addText("The working app, answering a question on City rents",
            box({ x: shotX, y: shotY + shotH + 0.08, w: shotW, h: 0.25, fontSize: 10, italic: true,
                  color: C.background2, objectName: "Screenshot caption" }));
  s.addNotes("This is the London Office Monitor, a working proof of concept for the London office team. Think of it " +
             "as a research assistant. It reads broker figures, property news and insolvency notices, keeps everything " +
             "it reads in one place, and points out what matters to Nan Fung. The team uses it three ways: ask it a " +
             "question, read the weekly brief it drafts, and discover opportunities that fit our mandate, which the " +
             "analyst approves before the team sees them. On the right " +
             "is the real app answering a question about City rents, with every broker's figure and a link to each " +
             "source. The next three slides cover what it does for the team: time, data and opportunities.");

  // ---------- 2. Time saving ----------
  pres.addSection({ title: "How it helps" });
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "How it helps" });
  s.addText("1 · TIME SAVING", { placeholder: "kicker", charSpacing: 1 });
  s.addText("Routine research now takes minutes", { placeholder: "title" });
  const tasks = [
    { time: TIMES.answer, unit: "per answer", head: "Compare the brokers on any figure",
      today: "Open six broker reports, find the figure, check each definition",
      now: "One table with every broker's figure, each linked to its source" },
    { time: TIMES.brief, unit: "per brief", head: "Write the weekly market brief",
      today: "Read the news across many outlets, then write it up",
      now: "A first draft in a set format, with every figure checked" },
    { time: TIMES.scan, unit: "per scan", head: "Screen the week for opportunities",
      today: "Read news and insolvency notices against the mandate by hand",
      now: "Up to five leads investigated and written up as cards" },
  ];
  for (let i = 0; i < tasks.length; i++) {
    const y = 1.5 + i * 0.98;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y, w: 9, h: 0.88, rectRadius: 0.08,
                                                fill: { color: C.background2 }, line: { color: C.background2 },
                                                objectName: tasks[i].head + " card" });
    s.addText(tasks[i].time, box({ x: 0.7, y: y + 0.1, w: 1.6, h: 0.48, fontSize: 28, bold: true, color: C.accent1,
                                   fontFace: THEME.headFontFace, valign: "middle", objectName: "Time" }));
    s.addText(tasks[i].unit, box({ x: 0.7, y: y + 0.58, w: 1.6, h: 0.22, fontSize: 11, color: C.accent6 }));
    s.addText(tasks[i].head, box({ x: 2.45, y: y + 0.08, w: 6.85, h: 0.3, fontSize: 16, bold: true, color: C.text2 }));
    s.addText([
      { text: "Today  ", options: { bold: true, color: C.accent6 } },
      { text: tasks[i].today, options: { color: C.accent6, breakLine: true } },
      { text: "Now  ", options: { bold: true, color: C.accent2 } },
      { text: tasks[i].now, options: { color: C.text1 } },
    ], box({ x: 2.45, y: y + 0.36, w: 6.85, h: 0.5, fontSize: 14 }));
  }
  s.addImage({ data: await icon(fa.FaCoins, HEX.accent4), x: 0.55, y: 4.45, w: 0.26, h: 0.26, altText: "Coins icon" });
  s.addText("Each one costs a few US cents of AI usage, under US$0.10.",
            box({ x: 0.95, y: 4.43, w: 8.5, h: 0.3, fontSize: 14, color: C.text1, valign: "middle" }));
  footnote(s, C, TIMES_NOTE);
  s.addNotes("First, time. To compare what six brokers say about one figure, an analyst opens six reports, finds the " +
             "figure in each, and checks how each broker defines it. The Monitor does it in about 15 seconds and links " +
             "every figure to its source. The weekly brief takes about a minute to draft, and the opportunity scan " +
             "about two. Each costs a few cents of AI usage. These times come from our test runs on 24 and 25 October. " +
             "We have not yet measured how long the same work takes the team by hand, so that comparison is still to " +
             "come.");

  // ---------- 3. Long-term data management ----------
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "How it helps" });
  s.addText("2 · LONG-TERM DATA MANAGEMENT", { placeholder: "kicker", charSpacing: 1 });
  s.addText("A market memory that stays with the team", { placeholder: "title" });
  const keeps = [
    [fa.FaChartBar, "Broker figures", "Rents, vacancy, yields and more"],
    [fa.FaNewspaper, "News and notices", "Each story with its link and date"],
    [fa.FaFolderOpen, "Our own records", "Policies, committee notes, decisions"],
    [fa.FaClipboardCheck, "The team's calls", "Every approve or reject, and why"],
  ];
  const matters = [
    [fa.FaEquals, "Same question, same number", "One shared record for the whole team"],
    [fa.FaLink, "Every figure is traceable", "One click opens the original source"],
    [fa.FaUserFriends, "Knowledge stays with the team", "Past decisions stay searchable"],
    [fa.FaSeedling, "It grows in value", "History builds; new sources plug in"],
  ];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 1.5, w: 4.3, h: 3.25, rectRadius: 0.08,
                                              fill: { color: C.text2 }, line: { color: C.text2 },
                                              objectName: "What it keeps panel" });
  s.addText("WHAT IT KEEPS, IN OUR OWN DATABASE", box({ x: 0.75, y: 1.65, w: 3.9, h: 0.28, fontSize: 12, bold: true,
                                                        color: C.accent4, charSpacing: 1 }));
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.5, w: 4.3, h: 3.25, rectRadius: 0.08,
                                              fill: { color: C.background2 }, line: { color: C.background2 },
                                              objectName: "Why it matters panel" });
  s.addText("WHY IT MATTERS", box({ x: 5.45, y: 1.65, w: 3.9, h: 0.28, fontSize: 12, bold: true, color: C.accent1,
                                    charSpacing: 1 }));
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 4.88, y: 2.95, w: 0.24, h: 0.32, fill: { color: C.accent3 },
                                        line: { color: C.accent3 }, objectName: "Arrow" });
  for (let i = 0; i < keeps.length; i++) {
    const y = 2.1 + i * 0.64;
    s.addImage({ data: await icon(keeps[i][0], HEX.accent4), x: 0.78, y: y + 0.08, w: 0.3, h: 0.3,
                 altText: keeps[i][1] + " icon" });
    s.addText(keeps[i][1], box({ x: 1.25, y, w: 3.45, h: 0.26, fontSize: 14, bold: true, color: C.background1 }));
    s.addText(keeps[i][2], box({ x: 1.25, y: y + 0.26, w: 3.45, h: 0.26, fontSize: 14, color: C.background2 }));
    s.addShape(pres.shapes.OVAL, { x: 5.45, y: y + 0.02, w: 0.42, h: 0.42, fill: { color: C.text2 },
                                   line: { color: C.text2 }, objectName: matters[i][1] + " icon circle" });
    s.addImage({ data: await icon(matters[i][0], HEX.lt1), x: 5.56, y: y + 0.13, w: 0.2, h: 0.2,
                 altText: matters[i][1] + " icon" });
    s.addText(matters[i][1], box({ x: 6.05, y, w: 3.35, h: 0.26, fontSize: 14, bold: true, color: C.text2 }));
    s.addText(matters[i][2], box({ x: 6.05, y: y + 0.26, w: 3.35, h: 0.26, fontSize: 14, color: C.text1 }));
  }
  footnote(s, C, "Broker figures come from JLL, Savills, BNP Paribas, Cushman & Wakefield, CBRE and Avison Young. In the " +
                 "demo, the internal records are mock examples; real ones need an access review before loading.");
  s.addNotes("Second, data. Everything the Monitor reads is kept in our own database: broker figures with their " +
             "definitions, news and insolvency notices with their links, our own policies and past decisions, and " +
             "every approve or reject call with its reason. That gives the team one shared record. The same question " +
             "gets the same number, every figure opens its original source, and past decisions stay searchable when " +
             "people move on. The history grows as each quarter's figures are added, and licensed data sources can be " +
             "added later without rebuilding it. In the demo, the internal documents are mock examples.");

  // ---------- 4. Opportunity discovery ----------
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "How it helps" });
  s.addText("3 · OPPORTUNITY DISCOVERY", { placeholder: "kicker", charSpacing: 1 });
  s.addText("Spots opportunities that fit Nan Fung", { placeholder: "title" });
  s.addText("Most London office deals trade off-market, so hearing early matters.",
            box({ x: 0.5, y: 1.4, w: 9, h: 0.3, fontSize: 15, italic: true, color: C.accent6, objectName: "Why early" }));
  const steps = [
    ["Reads", "The week's news and insolvency notices"],
    ["Matches", "Against our mandate and our buildings"],
    ["Investigates", "Up to five leads, each double-checked"],
    ["You decide", "Only approved cards reach the team"],
  ];
  for (let i = 0; i < steps.length; i++) {
    const y = 1.95 + i * 0.7;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y, w: 0.42, h: 0.42, fill: { color: C.accent1 }, line: { color: C.accent1 },
                                   objectName: "Step " + (i + 1) + " circle" });
    s.addText(String(i + 1), box({ x: 0.5, y, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: C.background1,
                                   align: "center", valign: "middle" }));
    s.addText(steps[i][0], box({ x: 1.07, y: y - 0.03, w: 3.75, h: 0.27, fontSize: 15, bold: true, color: C.text2 }));
    s.addText(steps[i][1], box({ x: 1.07, y: y + 0.24, w: 3.75, h: 0.27, fontSize: 14, color: C.text1 }));
  }
  const card = { x: 4.95, y: 1.85, w: 4.55, h: 2.92 };
  const cx = card.x + 0.22, cw = card.w - 0.44;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { ...card, rectRadius: 0.06, fill: { color: C.background1 },
                                              line: { color: C.background2, width: 1 }, shadow: softShadow(),
                                              objectName: "Example opportunity card" });
  s.addText("EXAMPLE FROM THE DEMO · TEST DATA", box({ x: cx, y: card.y + 0.14, w: cw, h: 0.22, fontSize: 10,
                                                       bold: true, color: C.accent6, charSpacing: 1 }));
  // The card as the web app shows it (ui/view.py card_markdown): grey tags, then what it means and what to do next.
  const tags = [["Owner under pressure", 1.75], ["City Core", 0.9]];
  for (let i = 0, tagX = cx; i < tags.length; tagX += tags[i][1] + 0.1, i++) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: tagX, y: card.y + 0.44, w: tags[i][1], h: 0.26, rectRadius: 0.05,
                                                fill: { color: C.background2 }, line: { color: C.background2 },
                                                objectName: tags[i][0] + " tag" });
    s.addText(tags[i][0], box({ x: tagX, y: card.y + 0.44, w: tags[i][1], h: 0.26, fontSize: 11, bold: true,
                                color: C.accent6, align: "center", valign: "middle" }));
  }
  s.addText(`Administrators appointed to the owner of a 45,000${NBSP}sq${NBSP}ft City office`,
            box({ x: cx, y: card.y + 0.8, w: cw, h: 0.5, fontSize: 14, bold: true, color: C.text2 }));
  s.addText([
    { text: "What it means for us: ", options: { bold: true } },
    { text: "bought in 2018 on a five-year loan; fits our City mandate" },
  ], box({ x: cx, y: card.y + 1.38, w: cw, h: 0.44, fontSize: 12, color: C.text1 }));
  s.addText([
    { text: "What to do next: ", options: { bold: true } },
    { text: "confirm at Companies House and contact the administrator" },
  ], box({ x: cx, y: card.y + 1.9, w: cw, h: 0.44, fontSize: 12, color: C.text1 }));
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: card.y + 2.48, w: 1.1, h: 0.28, rectRadius: 0.05,
                                              fill: { color: C.text2 }, line: { color: C.text2 },
                                              objectName: "Approve button" });
  s.addText("Approve", box({ x: cx, y: card.y + 2.48, w: 1.1, h: 0.28, fontSize: 12, bold: true, color: C.background1,
                             align: "center", valign: "middle" }));
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx + 1.25, y: card.y + 2.48, w: 1.1, h: 0.28, rectRadius: 0.05,
                                              fill: { color: C.background1 }, line: { color: C.accent3, width: 1 },
                                              objectName: "Reject button" });
  s.addText("Reject", box({ x: cx + 1.25, y: card.y + 2.48, w: 1.1, h: 0.28, fontSize: 12, color: C.text1,
                            align: "center", valign: "middle" }));
  footnote(s, C, "Off-market share: 69% of City deals (H1 2019, Bisnow) and 76% of West End volume (July 2022, " +
                 "Savills). The example is built on a test notice and an illustrative mandate.");
  s.addNotes("Third, opportunities. Most London office deals trade off-market, so hearing early matters. The scan reads " +
             "the week's news and insolvency notices and matches them against our mandate and our buildings. It " +
             "investigates up to five leads at once, and a second AI model checks each card against its evidence. The " +
             "analyst approves or rejects each card, and only approved cards reach the team's Approved list. The scan also checks " +
             "past decisions and tries to skip leads the analyst already rejected. It can be scheduled to run by itself " +
             "every Monday morning. The card on the right comes from the demo: a test insolvency notice matched " +
             "against an illustrative mandate, since Nan Fung's real criteria are not public.");

  // ---------- 5. Why not just use ChatGPT ----------
  pres.addSection({ title: "Trust and the app" });
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Trust and the app" });
  s.addText("BUILT TO BE TRUSTED", { placeholder: "kicker", charSpacing: 1 });
  s.addText("Why not just use ChatGPT?", { placeholder: "title" });
  const cols = [{ x: 0.5, w: 2.3 }, { x: 2.9, w: 2.7 }, { x: 5.7, w: 3.8 }];
  const rows = [
    ["Market figures", "What it finds or recalls", "Checked broker figures, with sources"],
    ["Numbers checked first", "Not guaranteed", "Always, against our stored data"],
    ["Knows Nan Fung", "Only what you give it", "Our buildings, mandate and decisions"],
    ["Team sign-off", "None", "The analyst approves each lead"],
    ["Team memory", "Personal memory only", "Past decisions shape the next scan"],
    ["Control over sources", "Reads any site it finds", "A fixed list we control"],
  ];
  s.addShape(pres.shapes.RECTANGLE, { x: cols[1].x, y: 1.45, w: cols[1].w, h: 0.4, fill: { color: C.background2 },
                                      line: { color: C.background2 }, objectName: "Chatbot header" });
  s.addText("ChatGPT", box({ x: cols[1].x + 0.15, y: 1.45, w: cols[1].w - 0.3, h: 0.4, fontSize: 14,
                                          bold: true, color: C.accent6, valign: "middle" }));
  s.addShape(pres.shapes.RECTANGLE, { x: cols[2].x, y: 1.45, w: cols[2].w, h: 0.4, fill: { color: C.text2 },
                                      line: { color: C.text2 }, objectName: "Monitor header" });
  s.addText("London Office Monitor", box({ x: cols[2].x + 0.15, y: 1.45, w: cols[2].w - 0.3, h: 0.4, fontSize: 14,
                                           bold: true, color: C.background1, valign: "middle" }));
  const cross = await icon(fa.FaTimes, HEX.accent3), tick = await icon(fa.FaCheck, HEX.accent2);
  for (let i = 0; i < rows.length; i++) {
    const y = 1.93 + i * 0.5, h = 0.44, markY = y + (h - 0.18) / 2;
    s.addText(rows[i][0], box({ x: cols[0].x, y, w: cols[0].w, h, fontSize: 14, bold: true, color: C.text2,
                                valign: "middle" }));
    s.addShape(pres.shapes.RECTANGLE, { x: cols[1].x, y, w: cols[1].w, h, fill: { color: C.background2 },
                                        line: { color: C.background2 }, objectName: rows[i][0] + " chatbot cell" });
    s.addImage({ data: cross, x: cols[1].x + 0.15, y: markY, w: 0.18, h: 0.18, altText: "No" });
    s.addText(rows[i][1], box({ x: cols[1].x + 0.45, y, w: cols[1].w - 0.55, h, fontSize: 14, color: C.accent6,
                                valign: "middle" }));
    s.addShape(pres.shapes.RECTANGLE, { x: cols[2].x, y, w: cols[2].w, h, fill: { color: C.accent2, transparency: 88 },
                                        line: { color: C.accent2, transparency: 88 },
                                        objectName: rows[i][0] + " monitor cell" });
    s.addImage({ data: tick, x: cols[2].x + 0.15, y: markY, w: 0.18, h: 0.18, altText: "Yes" });
    s.addText(rows[i][2], box({ x: cols[2].x + 0.45, y, w: cols[2].w - 0.55, h, fontSize: 14, color: C.text1,
                                valign: "middle" }));
  }
  s.addNotes("The obvious question: why not just use ChatGPT? ChatGPT can search the web and cite sources, but any " +
             "checking is the AI judging its own work, so nothing guarantees that each number matches its source. It " +
             "doesn't hold our broker data, knows only what you give it, and has no approval step before something " +
             "reaches the team. In the Monitor, every figure in an answer is checked against our stored data before " +
             "you see it; a mismatch is rewritten or shown with a visible warning. It knows our buildings and mandate, " +
             "and the analyst approves every lead before the team sees it. Those decisions are kept with their reasons, " +
             "and each new scan checks them, so a lead like one the analyst turned down is dropped or explained. " +
             "ChatGPT's memory, by contrast, is personal to each user. It also works from a fixed list of sources " +
             "we control, so we can leave out any whose terms forbid AI use, such as Knight Frank, and broker reports " +
             "are never fed to the AI.");

  // ---------- 6. A quick tour of the app (dark) ----------
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Trust and the app" });
  s.addText("IN THE APP", { placeholder: "kicker", charSpacing: 1 });
  s.addText("A quick tour of the app", { placeholder: "title" });
  for (let i = 0; i < TOUR.length; i++) {
    const stop = TOUR[i], tourShot = await screenshot(stop);
    const x = 0.5 + i * 3.08, y = 1.55, w = 2.84, h = w / tourShot.aspect;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background1 }, line: { color: C.background1 },
                                        shadow: softShadow(), objectName: stop.head + " screenshot frame" });
    s.addImage({ data: tourShot.data, x, y, w, h, altText: stop.alt });
    s.addImage({ data: await icon(stop.icon, HEX.accent4), x, y: y + h + 0.22, w: 0.3, h: 0.3,
                 altText: stop.head + " icon" });
    s.addText(stop.head, box({ x: x + 0.42, y: y + h + 0.18, w: w - 0.42, h: 0.38, fontSize: 17, bold: true,
                               color: C.background1, valign: "middle" }));
    s.addText(stop.text, box({ x, y: y + h + 0.62, w, h: 0.55, fontSize: 14, color: C.background2 }));
  }
  s.addNotes("To finish, a quick look at the app itself. In Chat, you ask a question in plain words; here, how City " +
             "prime rent has moved, and every broker's figure comes back on a chart with a link to each source. The " +
             "weekly brief is one click: it drafts what changed, the news themes, rates and what it means for our " +
             "buildings. Opportunities is where the scan's leads arrive: the analyst approves or rejects each one under " +
             "To review, and approved ones are shared with the team under Approved. Everything shown is the demo " +
             "snapshot from 24 October 2026, and the card shown is marked as a demo example.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
}

main().catch(e => { console.error(e); process.exit(1); });
