'use client';

import { useState } from 'react';
import type { MakeVideoProgress } from '@/lib/agent';
import type { ProjectView, RunStatus } from '@/lib/studio';

export function PlanCard({ view, heading }: { view: ProjectView; heading: string }) {
  const { plan, pending_image_calls: pending } = view;
  return (
    <div className="card">
      <div className="card-eyebrow">{heading}</div>
      <div className="card-title">{plan.title}</div>
      <dl className="plan-meta">
        <dt>Actor</dt>
        <dd>
          {plan.actor.name} <span className="tag">{plan.actor.kind.toLowerCase()}</span>
        </dd>
        <dt>Style</dt>
        <dd>{plan.style}</dd>
      </dl>
      <ol className="route">
        {plan.scenes.map(s => (
          <li key={s.name} title={s.visual}>
            {s.name}
          </li>
        ))}
      </ol>
      <div className="card-foot">
        {pending.length === 0
          ? 'All images ready. Re-rendering is free.'
          : `${pending.length} image${pending.length === 1 ? '' : 's'} to generate`}
        {view.removed && view.removed.length > 0 && ` · ${view.removed.length} old image(s) discarded`}
      </div>
    </div>
  );
}

export function ApprovalCard({
  reason,
  onApprove,
  onDeny,
}: {
  reason?: string;
  onApprove: () => void;
  onDeny: () => void;
}) {
  // reason: "N new images will be generated: actor: X; scene: Y; ..." (items contain ': ' too)
  const text = reason ?? 'New images will be generated';
  const cut = text.indexOf(': ');
  const summary = cut < 0 ? text : text.slice(0, cut);
  const list = cut < 0 ? '' : text.slice(cut + 2);
  return (
    <div className="card approval">
      <div className="card-eyebrow">Needs your approval</div>
      <div className="card-title">{summary}</div>
      {list && (
        <ul className="pending-list">
          {list.split('; ').map(item => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}
      <p className="muted">These use your OpenAI image credits.</p>
      <div className="actions">
        <button type="button" className="primary" onClick={onApprove}>
          Approve and make video
        </button>
        <button type="button" onClick={onDeny}>
          Not now
        </button>
      </div>
    </div>
  );
}

export function ProgressLine({ step }: { step: string }) {
  return (
    <div className="progress" role="status">
      <span className="spinner" aria-hidden />
      {step}
    </div>
  );
}

export function VideoCard({ videos, duration }: { videos: NonNullable<RunStatus['videos']>; duration?: number }) {
  const formats = (['16:9', '9:16'] as const).filter(f => videos[f]);
  const [format, setFormat] = useState<(typeof formats)[number]>(formats[0] ?? '16:9');
  const src = videos[format];
  if (!src) return null;
  return (
    <div className="card video-card">
      <div className="video-head">
        <div className="segmented" role="tablist">
          {formats.map(f => (
            <button key={f} type="button" role="tab" aria-selected={f === format} onClick={() => setFormat(f)}>
              {f}
            </button>
          ))}
        </div>
        {duration !== undefined && <span className="muted">{Math.round(duration)}s</span>}
        <a className="download" href={src} download>
          Download
        </a>
      </div>
      {/* Cache-bust so a re-render of the same project shows the new file */}
      <video key={src} className={format === '9:16' ? 'vertical' : undefined} controls autoPlay muted loop playsInline>
        <source src={`${src}?v=${duration ?? ''}`} type="video/mp4" />
      </video>
    </div>
  );
}

export function MakeVideoResult({ output }: { output: MakeVideoProgress }) {
  if (output.state === 'running') return <ProgressLine step={output.step} />;
  if (output.state === 'failed') return <div className="card error">Render failed: {output.error}</div>;
  return <VideoCard videos={output.videos} duration={output.duration} />;
}
