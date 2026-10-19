// Typed client for the Python server's /api/make routes (workshop/server/make_routes.py).

const STUDIO_URL = process.env.STUDIO_API_URL ?? 'http://127.0.0.1:8000';

export const GROUND_MATERIALS = ['ASPHALT', 'GRASS', 'WOODEN_FLOOR', 'WATER_PUDDLE', 'SNOW', 'STONE'] as const;
export const AMBIENT_MOODS = [
  'RAINY_LOFI',
  'NIGHT_CRICKETS',
  'FOREST_WIND',
  'COZY_FIREPLACE',
  'CYBERPUNK_HUM',
  'OCEAN_WAVES',
  'SILENT',
] as const;
export const ACTOR_KINDS = ['WHEELED', 'FLOATING', 'WALKER'] as const;

export type PlannedScene = {
  name: string;
  visual: string;
  ground_material: (typeof GROUND_MATERIALS)[number];
  ambient_mood: (typeof AMBIENT_MOODS)[number];
};

export type JourneyPlan = {
  prompt: string;
  title: string;
  style: string;
  actor: { name: string; kind: (typeof ACTOR_KINDS)[number]; visual: string };
  scenes: PlannedScene[];
};

export type RunStatus = {
  status: 'NOT_STARTED' | 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  step?: string;
  error?: string;
  videos?: { '16:9': string | null; '9:16': string | null };
  duration?: number;
};

export type ProjectView = {
  project_id: string;
  plan: JourneyPlan;
  scene_ids: string[];
  pending_image_calls: string[];
  options: { seconds_per_scene: number; actor_size: number };
  status: RunStatus | null;
  removed?: string[];
};

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${STUDIO_URL}/api/make${path}`, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
  });
  if (!res.ok) {
    const detail = await res.json().then(
      (j: { detail?: unknown }) => (typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)),
      () => res.statusText,
    );
    throw new Error(`Studio API ${method} ${path} failed (${res.status}): ${detail}`);
  }
  return (await res.json()) as T;
}

export const studio = {
  createPlan: (prompt: string, numScenes: number) =>
    call<ProjectView>('POST', '/plan', { prompt, num_scenes: numScenes }),
  getProject: (id: string) => call<ProjectView>('GET', `/${id}`),
  updatePlan: (id: string, plan: Omit<JourneyPlan, 'prompt'>) => call<ProjectView>('PUT', `/${id}/plan`, plan),
  redo: (id: string, target: string) => call<ProjectView>('POST', `/${id}/redo`, { target }),
  updateOptions: (id: string, options: { seconds_per_scene?: number; actor_size?: number }) =>
    call<ProjectView>('PUT', `/${id}/options`, options),
  run: (id: string) => call<ProjectView>('POST', `/${id}/run`),
  status: (id: string) => call<RunStatus>('GET', `/${id}/status`),
};
