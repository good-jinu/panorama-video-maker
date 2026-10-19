'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport, lastAssistantMessageIsCompleteWithApprovalResponses } from 'ai';
import { type FormEvent, useEffect, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import { ApprovalCard, MakeVideoResult, PlanCard, ProgressLine } from '@/components/cards';
import type { VideoAgentUIMessage } from '@/lib/agent';

const SUGGESTIONS = [
  "A small car riding through the world's cities",
  'A red hot-air balloon drifting over European capitals',
  'A tiny steam train crossing the four seasons',
];

type Part = VideoAgentUIMessage['parts'][number];

export default function ChatPage() {
  const { messages, sendMessage, addToolApprovalResponse, status, stop, error } = useChat<VideoAgentUIMessage>({
    transport: new DefaultChatTransport({ api: '/api/chat' }),
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses,
  });
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const busy = status === 'submitted' || status === 'streaming';

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  const send = (text: string) => {
    if (!text.trim() || busy) return;
    sendMessage({ text: text.trim() });
    setInput('');
  };
  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    send(input);
  };

  const renderPart = (part: Part, key: string, role: VideoAgentUIMessage['role']) => {
    switch (part.type) {
      case 'text':
        if (!part.text) return null;
        // User text stays literal; assistant replies are markdown (raw HTML is not rendered)
        return (
          <div key={key} className="text">
            {role === 'user' ? part.text : <Markdown>{part.text}</Markdown>}
          </div>
        );

      case 'tool-planVideo':
      case 'tool-updatePlan':
      case 'tool-getProject':
      case 'tool-redoImage':
      case 'tool-setRenderOptions': {
        if (part.state === 'output-available') {
          const view = part.output;
          switch (part.type) {
            case 'tool-getProject':
              return null;
            case 'tool-setRenderOptions':
              return (
                <div key={key} className="muted note">
                  Render settings: {view.options.seconds_per_scene}s per scene · actor size ×{view.options.actor_size}
                </div>
              );
            case 'tool-redoImage':
              return (
                <div key={key} className="muted note">
                  Discarded {view.removed?.join(', ') || 'nothing'} · {view.pending_image_calls.length} image(s) to
                  generate next time
                </div>
              );
            default:
              return <PlanCard key={key} view={view} heading={part.type === 'tool-planVideo' ? 'Plan' : 'Updated plan'} />;
          }
        }
        if (part.state === 'output-error') return <div key={key} className="card error">{part.errorText}</div>;
        const label = part.type === 'tool-planVideo' ? 'Planning the journey...' : 'Updating...';
        return <ProgressLine key={key} step={label} />;
      }

      case 'tool-makeVideo':
        switch (part.state) {
          case 'input-streaming':
          case 'input-available':
            return <ProgressLine key={key} step="Checking what needs to be generated..." />;
          case 'approval-requested':
            if (part.approval.isAutomatic) return null;
            return (
              <ApprovalCard
                key={key}
                reason={part.approval.requestReason}
                onApprove={() => addToolApprovalResponse({ id: part.approval.id, approved: true })}
                onDeny={() => addToolApprovalResponse({ id: part.approval.id, approved: false })}
              />
            );
          case 'approval-responded':
            return (
              <ProgressLine key={key} step={part.approval.approved ? 'Approved. Starting...' : 'Cancelled.'} />
            );
          case 'output-available':
            return <MakeVideoResult key={key} output={part.output} />;
          case 'output-denied':
            return (
              <div key={key} className="muted note">
                Not generated.
              </div>
            );
          case 'output-error':
            return <div key={key} className="card error">{part.errorText}</div>;
        }
        return null;

      default:
        return null;
    }
  };

  return (
    <div className="shell">
      <header className="topbar">
        <span className="logo" aria-hidden>
          ◐
        </span>
        <span className="brand">Panorama Video Maker</span>
      </header>

      <main className="thread">
        {messages.length === 0 && (
          <section className="empty">
            <h1>What journey should we film?</h1>
            <p className="muted">
              Describe an actor and where it travels. I&apos;ll plan the scenes, show you the cost, and make one
              continuous side-scrolling video.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map(s => (
                <button key={s} type="button" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </section>
        )}

        {messages.map(m => (
          <div key={m.id} className={`msg ${m.role}`}>
            {m.parts.map((part, i) => renderPart(part, `${m.id}-${i}`, m.role))}
          </div>
        ))}

        {status === 'submitted' && <ProgressLine step="Thinking..." />}
        {error && <div className="card error">Something went wrong: {error.message}</div>}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={onSubmit}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={messages.length === 0 ? 'A small car riding through world cities...' : 'Ask for changes...'}
          aria-label="Message"
          autoFocus
        />
        {busy ? (
          <button type="button" onClick={() => stop()}>
            Stop
          </button>
        ) : (
          <button type="submit" className="primary" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
}
