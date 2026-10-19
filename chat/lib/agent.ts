import { google } from '@ai-sdk/google';
import { type InferAgentUIMessage, ToolLoopAgent, tool } from 'ai';
import { randomBytes } from 'node:crypto';
import { z } from 'zod';
import { ACTOR_KINDS, AMBIENT_MOODS, GROUND_MATERIALS, type ProjectView, type RunStatus, studio } from './studio';

const projectId = z.string().describe('The project_id returned by planVideo');

const sceneSchema = z.object({
  name: z.string().describe('Unique place name, e.g. "Tokyo"'),
  visual: z.string().describe('What the backdrop looks like. Never mention the actor, vehicles or people.'),
  ground_material: z.enum(GROUND_MATERIALS),
  ambient_mood: z.enum(AMBIENT_MOODS),
});

const POLL_MS = 1500;
const RUN_TIMEOUT_MS = 45 * 60 * 1000;

export type MakeVideoProgress =
  | { state: 'running'; step: string }
  | { state: 'done'; projectId: string; videos: NonNullable<RunStatus['videos']>; duration?: number }
  | { state: 'failed'; error: string };

const tools = {
  planVideo: tool({
    description:
      'Plan a NEW video from the user idea: actor, art style and an ordered list of scenes. ' +
      'Makes no images and costs almost nothing. Use once per new idea; use updatePlan to change an existing plan.',
    inputSchema: z.object({
      prompt: z.string().describe("The user's idea, in their words"),
      numScenes: z.number().int().min(2).max(8).default(5),
    }),
    execute: async ({ prompt, numScenes }): Promise<ProjectView> => studio.createPlan(prompt, numScenes),
  }),

  getProject: tool({
    description: 'Read the current plan, pending image calls, render options and last run status of a project.',
    inputSchema: z.object({ projectId }),
    execute: async ({ projectId }): Promise<ProjectView> => studio.getProject(projectId),
  }),

  updatePlan: tool({
    description:
      'Replace the whole plan of an existing project (send every scene, in travel order). ' +
      'Only images whose description changed are regenerated on the next makeVideo. Makes no images itself.',
    inputSchema: z.object({
      projectId,
      title: z.string(),
      style: z.string().describe('One art style for every image. Changing it regenerates everything.'),
      actor: z.object({ name: z.string(), kind: z.enum(ACTOR_KINDS), visual: z.string() }),
      scenes: z.array(sceneSchema).min(2).max(8),
    }),
    execute: async ({ projectId, ...plan }): Promise<ProjectView> => studio.updatePlan(projectId, plan),
  }),

  redoImage: tool({
    description:
      'Discard one generated image so the next makeVideo makes a new version: "actor", or a scene id from scene_ids ' +
      "(this also redoes that scene's bridges). Use when the user dislikes an image but keeps its description.",
    inputSchema: z.object({ projectId, target: z.string() }),
    execute: async ({ projectId, target }): Promise<ProjectView> => studio.redo(projectId, target),
  }),

  setRenderOptions: tool({
    description:
      'Change render settings: secondsPerScene (higher is slower, default 12) and actorSize (multiplier, default 1). ' +
      'Free: no new images. Call makeVideo afterwards to re-render.',
    inputSchema: z.object({
      projectId,
      secondsPerScene: z.number().min(2).max(120).optional(),
      actorSize: z.number().min(0.3).max(3).optional(),
    }),
    execute: async ({ projectId, secondsPerScene, actorSize }): Promise<ProjectView> =>
      studio.updateOptions(projectId, { seconds_per_scene: secondsPerScene, actor_size: actorSize }),
  }),

  makeVideo: tool({
    description:
      'Generate any missing images and render the video (16:9 and 9:16). Can take several minutes. ' +
      'The user is asked to approve first whenever new images will be generated.',
    inputSchema: z.object({ projectId }),
    async *execute({ projectId }): AsyncGenerator<MakeVideoProgress> {
      await studio.run(projectId);
      yield { state: 'running', step: 'Starting...' };

      const deadline = Date.now() + RUN_TIMEOUT_MS;
      while (Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, POLL_MS));
        const st = await studio.status(projectId);
        if (st.status === 'COMPLETED' && st.videos) {
          yield { state: 'done', projectId, videos: st.videos, duration: st.duration };
          return;
        }
        if (st.status === 'FAILED') {
          yield { state: 'failed', error: st.error ?? 'Unknown error' };
          return;
        }
        yield { state: 'running', step: st.step ?? 'Working...' };
      }
      yield { state: 'failed', error: 'Timed out waiting for the render' };
    },
  }),
};

export const videoAgent = new ToolLoopAgent({
  model: google(process.env.CHAT_MODEL ?? 'gemini-flash-latest'),
  instructions: `You are the director inside Panorama Video Maker. It makes side-scrolling videos where the camera
follows one actor travelling left-to-right through a sequence of scenes that are stitched into one continuous panorama.

How to work:
- New idea: call planVideo. The user already sees the plan as a card (title, actor, style, scenes, image count), so do
  not repeat it: add at most one sentence of your own, then ask if they want to make it or change something.
  Do not call makeVideo before the user agrees.
- Changes: call updatePlan with the full revised plan, setRenderOptions for speed or actor size, or redoImage to get a
  new version of an image with the same description. Then call makeVideo when the user wants to see the result.
- Always reuse the project_id from earlier tool results. Never invent one.
- Supported actors roll (cars, bikes, trains) or float (balloons, boats, planes). Walking characters are not supported
  yet; offer a rolling or floating alternative instead.
- If makeVideo is not approved, do not retry it; ask what to change instead.
- When a video is done, say so in one line. The video player is shown to the user automatically.
Keep replies short and friendly. Reply in the user's language.`,
  tools,
  toolApproval: {
    // Spending gate: ask the user only when new images would be generated; pure re-renders run immediately.
    makeVideo: async ({ projectId }) => {
      const view = await studio.getProject(projectId);
      const pending = view.pending_image_calls;
      if (pending.length === 0) return undefined;
      return {
        type: 'user-approval',
        reason: `${pending.length} new image${pending.length === 1 ? '' : 's'} will be generated: ${pending.join('; ')}`,
      };
    },
  },
  // Binds approvals to this server process so a client cannot forge one from chat history.
  experimental_toolApprovalSecret: process.env.TOOL_APPROVAL_SECRET ?? randomBytes(32).toString('base64'),
});

export type VideoAgentUIMessage = InferAgentUIMessage<typeof videoAgent>;
