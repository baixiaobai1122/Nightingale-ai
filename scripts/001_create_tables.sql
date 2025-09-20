-- Create profiles table for user management
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  name text not null,
  role text not null check (role in ('doctor', 'patient')),
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create sessions table for medical consultations
create table if not exists public.sessions (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid not null references public.profiles(id) on delete cascade,
  doctor_id uuid references public.profiles(id) on delete set null,
  title text not null,
  status text not null default 'active' check (status in ('active', 'completed', 'cancelled')),
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create segments table for session recordings
create table if not exists public.segments (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.sessions(id) on delete cascade,
  speaker text not null,
  content text not null,
  timestamp_start integer not null,
  timestamp_end integer not null,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create summaries table for AI-generated summaries
create table if not exists public.summaries (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.sessions(id) on delete cascade,
  content text not null,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable Row Level Security
alter table public.profiles enable row level security;
alter table public.sessions enable row level security;
alter table public.segments enable row level security;
alter table public.summaries enable row level security;

-- Profiles policies
create policy "Users can view their own profile" on public.profiles
  for select using (auth.uid() = id);

create policy "Users can insert their own profile" on public.profiles
  for insert with check (auth.uid() = id);

create policy "Users can update their own profile" on public.profiles
  for update using (auth.uid() = id);

-- Sessions policies
create policy "Users can view their own sessions" on public.sessions
  for select using (auth.uid() = patient_id or auth.uid() = doctor_id);

create policy "Patients can create sessions" on public.sessions
  for insert with check (auth.uid() = patient_id);

create policy "Users can update their own sessions" on public.sessions
  for update using (auth.uid() = patient_id or auth.uid() = doctor_id);

-- Segments policies
create policy "Users can view segments of their sessions" on public.segments
  for select using (
    exists (
      select 1 from public.sessions 
      where sessions.id = segments.session_id 
      and (sessions.patient_id = auth.uid() or sessions.doctor_id = auth.uid())
    )
  );

create policy "Users can insert segments for their sessions" on public.segments
  for insert with check (
    exists (
      select 1 from public.sessions 
      where sessions.id = segments.session_id 
      and (sessions.patient_id = auth.uid() or sessions.doctor_id = auth.uid())
    )
  );

-- Summaries policies
create policy "Users can view summaries of their sessions" on public.summaries
  for select using (
    exists (
      select 1 from public.sessions 
      where sessions.id = summaries.session_id 
      and (sessions.patient_id = auth.uid() or sessions.doctor_id = auth.uid())
    )
  );

create policy "Users can insert summaries for their sessions" on public.summaries
  for insert with check (
    exists (
      select 1 from public.sessions 
      where sessions.id = summaries.session_id 
      and (sessions.patient_id = auth.uid() or sessions.doctor_id = auth.uid())
    )
  );
