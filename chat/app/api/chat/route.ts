import { createAgentUIStreamResponse } from 'ai';
import { videoAgent } from '@/lib/agent';

export const runtime = 'nodejs';

export async function POST(request: Request) {
  const { messages } = await request.json();
  return createAgentUIStreamResponse({
    agent: videoAgent,
    uiMessages: messages,
    abortSignal: request.signal,
  });
}
