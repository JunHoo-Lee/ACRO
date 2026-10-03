'use strict';

const datasets = {
  simpler: {
    title: 'GR00T N1.7 · SimplerEnv Bridge',
    columns: ['Task', 'Base', 'RoboMonkey†', 'CycleVLA†', 'ACRO'],
    rows: [['Carrot on Plate',50,31.7,48.3,68.3],['Eggplant in Basket',36.7,50,66.7,51.7],['Spoon on Towel',93.3,90,93.3,93.3],['Stack Cube',18.3,18.3,13.3,26.7]],
    overall: ['Overall',49.6,47.5,55.4,60],
    note: '300 physical steps per execution, including return motion. †Adapted baselines: RoboMonkey ranks policy samples with the shared Critic; CycleVLA uses a VLM for recovery. Details are in the paper.'
  },
  robocasa: {
    title: 'π₀.₅ · RoboCasa',
    columns: ['Task category', 'Base', 'RoboMonkey†', 'CycleVLA†', 'ACRO'],
    rows: [['Open / Close / Slide',64.4,66.8,65.6,76],['Pick and Place',64,68,56,65.2],['Control Operation',43,52,40,55]],
    overall: ['Overall',60.7,64.8,57.3,68],
    note: 'Each task’s registered physical step limit, including return motion. †Adapted baselines use the same frozen policy; implementation details are in the paper.'
  },
  real: {
    title: 'π₀.₅ · Franka Panda',
    columns: ['Task', 'Base', 'ACRO'],
    rows: [['Pick and Place',70,80],['Stack Cube',20,55],['Press Button',40,60]],
    overall: ['Overall',43.3,65],
    note: 'Pick and place, cube stacking, and button pressing with a fixed π₀.₅ policy. Physical return motion is included in the execution budget.'
  }
};

let activeResult = 'simpler';
let sortColumn = null;
let sortDirection = -1;
const resultTable = document.querySelector('#result-table');

function renderResults() {
  const data = datasets[activeResult];
  document.querySelector('#result-title').textContent = data.title;
  document.querySelector('#result-note').textContent = data.note;
  const head = document.createElement('tr');
  data.columns.forEach((label,index) => {
    const th = document.createElement('th');
    th.scope = 'col';
    th.setAttribute('aria-sort',sortColumn === index ? (sortDirection === 1 ? 'ascending' : 'descending') : 'none');
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.addEventListener('click',() => {sortDirection = sortColumn === index ? -sortDirection : (index === 0 ? 1 : -1);sortColumn = index;renderResults();resultTable.querySelectorAll('thead button')[index].focus();});
    th.append(button);head.append(th);
  });
  resultTable.tHead.replaceChildren(head);
  const rows = [...data.rows];
  if (sortColumn !== null) rows.sort((a,b) => typeof a[sortColumn] === 'number' ? sortDirection*(a[sortColumn]-b[sortColumn]) : sortDirection*a[sortColumn].localeCompare(b[sortColumn]));
  const renderedRows = [...rows,data.overall].map((row,index) => {
    const tr = document.createElement('tr');
    if (index === rows.length) tr.className = 'overall';
    const best = Math.max(...row.slice(1));
    row.forEach((value,col) => {
      const cell = document.createElement(col === 0 ? 'th' : 'td');
      if (col === 0) {cell.scope = 'row';cell.textContent = value;} else {
        cell.textContent = value.toFixed(1);
        if (value === best) cell.classList.add('best');
        if (col === row.length - 1) cell.classList.add('acro-cell');
      }
      tr.append(cell);
    });return tr;
  });
  resultTable.tBodies[0].replaceChildren(...renderedRows);
}

const tabs = [...document.querySelectorAll('[data-result]')];
function selectResult(tab) {
  activeResult = tab.dataset.result;sortColumn = null;
  tabs.forEach(t => {t.setAttribute('aria-selected',String(t === tab));t.tabIndex = t === tab ? 0 : -1;});
  document.querySelector('#result-panel').setAttribute('aria-labelledby',tab.id);
  renderResults();
}
tabs.forEach((tab,index) => {
  tab.addEventListener('click',() => selectResult(tab));
  tab.addEventListener('keydown',event => {
    const next = event.key === 'ArrowRight' ? (index+1)%tabs.length : event.key === 'ArrowLeft' ? (index+tabs.length-1)%tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length-1 : null;
    if (next !== null) {event.preventDefault();selectResult(tabs[next]);tabs[next].focus();}
  });
});
renderResults();

const video = document.querySelector('#rollout-video');
const terminal = document.querySelector('#terminal-frame');
const range = document.querySelector('#replay-step');
const play = document.querySelector('#play-video');

function updateStep(step) {
  step = Math.min(224,Math.max(0,Math.round(step)));
  range.value = step;document.querySelector('#step-label').textContent = step;
  terminal.hidden = step < 224;video.hidden = step >= 224;
  const label = document.querySelector('#phase-label');
  const text = document.querySelector('#phase-copy');
  if (step < 130) {
    label.textContent = 'Continue';label.style.background = '#2867b5';
    text.textContent = 'The frozen policy approaches the mixer. ACRO estimates the prospects of continuing with its proposed actions.';
  } else if (step < 180) {
    label.textContent = 'Retract';label.style.background = '#db493e';
    text.textContent = 'ACRO intervenes at step 130 and physically returns to an earlier configuration along a shortcut through visited space.';
  } else if (step < 224) {
    label.textContent = 'Resume';label.style.background = '#2867b5';
    text.textContent = 'At step 180, the VLA receives a fresh observation and begins a new approach. The policy weights stay fixed.';
  } else {
    label.textContent = 'Success';label.style.background = '#111';
    text.textContent = 'The stand mixer head opens at step 224: 50 return steps followed by 44 steps of renewed policy execution.';
  }
  const x = 26+step/224*386;
  document.querySelector('#q-cursor').setAttribute('x1',x);
  document.querySelector('#q-cursor').setAttribute('x2',x);
}
function seekStep(step) {
  video.pause();play.textContent = 'Play rollout';
  updateStep(step);
  if (Number.isFinite(video.duration)) video.currentTime = Math.min(step/20,Math.max(0,video.duration-.05));
}
range.addEventListener('input',() => seekStep(Number(range.value)));
document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click',() => seekStep(Number(button.dataset.step))));
play.addEventListener('click',async () => {
  if (video.paused) {
    if (Number(range.value) >= 224 || video.ended) {video.currentTime = 0;updateStep(0);}
    try {await video.play();play.textContent = 'Pause rollout';} catch {play.textContent = 'Play rollout';}
  } else {video.pause();play.textContent = 'Play rollout';}
});
video.addEventListener('timeupdate',() => {if (!video.paused && !video.seeking) updateStep(Math.floor(video.currentTime*20));});
video.addEventListener('ended',() => {updateStep(224);play.textContent = 'Play again';});

const svgNS = 'http://www.w3.org/2000/svg';
fetch('assets/data/recovery.json').then(r => {if (!r.ok) throw Error('Recovery data unavailable');return r.json();}).then(data => {
  const groups = [data.trace.filter(row => row.step <= 130),data.trace.filter(row => row.step >= 180)];
  groups.forEach(rows => {
    const path = document.createElementNS(svgNS,'polyline');
    path.setAttribute('points',rows.map(row => `${26+row.step/224*386},${141-row.q*127}`).join(' '));
    path.setAttribute('class','q-line');document.querySelector('#q-lines').append(path);
  });
}).catch(() => {document.querySelector('#q-chart').hidden = true;});

document.querySelector('#copy-citation').addEventListener('click',async () => {
  const text = document.querySelector('#bibtex').textContent;
  const button = document.querySelector('#copy-citation');
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = 'Copied';document.querySelector('#copy-status').textContent = 'Citation copied.';
    setTimeout(() => {button.textContent = 'Copy';},1800);
  } catch {
    const selection = window.getSelection();const selected = document.createRange();selected.selectNodeContents(document.querySelector('#bibtex'));selection.removeAllRanges();selection.addRange(selected);
    document.querySelector('#copy-status').textContent = 'Citation selected. Use your browser copy command.';
  }
});

const sectionLinks = [...document.querySelectorAll('.section-nav a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {entries.forEach(entry => {if (entry.isIntersecting) sectionLinks.forEach(link => link.classList.toggle('active',link.hash === `#${entry.target.id}`));});},{rootMargin:'-8% 0px -65% 0px',threshold:0});
  document.querySelectorAll('.article-section').forEach(section => observer.observe(section));
}
