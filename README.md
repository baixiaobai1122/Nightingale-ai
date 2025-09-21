# Nightingale AI - Medical Voice Assistant

A comprehensive medical AI system featuring privacy-first voice consultations with a **Next.js frontend** and **FastAPI backend** that demonstrates a **safe, provenance-grounded VoiceAI consultation workflow**.

##  Architecture

- **Frontend**: Next.js 15 with TypeScript, Tailwind CSS v4, and shadcn/ui components
- **Backend**: Hybrid architecture - Next.js API Routes + FastAPI backend (port 8000)
- **Database**: Supabase (PostgreSQL) with Row Level Security
- **Authentication**: Supabase Auth with role-based access control
- **AI Models**: External `omi-health/sum-small` model + local training infrastructure

*Note: Architecture alignment between frontend and backend is in progress.*

##  Key Features

###  Authentication & Authorization
- **Supabase Authentication**: Email/password authentication with role-based access
- **Role-Based Access Control**: Separate interfaces for patients and healthcare providers
- **Dynamic User Profiles**: User profiles with role-specific features and dynamic doctor names
- **Secure Session Management**: Automatic token refresh and validation

###  Medical Workflow Management
- **Session Tracking**: Unique session identifiers with status management
- **Voice Recording Integration**: Real-time audio capture and transcription
- **AI-Powered Summarization**: Generate both clinician and patient-friendly summaries
- **Human-in-the-Loop Review**: Doctor approval workflow for patient summaries
- **Patient Q&A System**: Search through approved medical records

###  Privacy & Security Features
- **PHI Redaction**: Automatic identification and redaction of Protected Health Information
- **Row Level Security**: Database-level access control for all medical data
- **Consent Management**: Explicit consent recording before AI processing
- **Audit Trails**: Complete logging of data access and modifications
- **HIPAA Compliance**: Healthcare data protection standards implementation
- **Data Encryption**: End-to-end encryption for all sensitive information

###  AI & Machine Learning
- **Intelligent Summarization**: Context-aware medical summary generation
- **Provenance Tracking**: Every AI-generated point links back to source conversation
- **Quality Assurance**: Grounding validation for medical accuracy
- **Dual Summary Types**: Technical summaries for clinicians, accessible summaries for patients

##  API Endpoints

*Note: APIs are split between Next.js routes (frontend) and FastAPI backend*

### Frontend API Routes (Next.js)
- `POST /api/session/start` - Initialize session (Supabase integration)
- `POST /api/summarize/[sessionId]` - Generate summaries (currently mock data)

### Backend-Proxied API Routes (FastAPI via Next.js)
- `POST /api/asr/ingest` - Ingest speech recognition segments
- `POST /api/consent/record` - Record patient consent with PHI redaction
- `POST /api/review/[summaryId]/approve` - Doctor approval workflow
- `GET /api/summary/[summaryId]` - Retrieve medical summary
- `GET /api/patient/[patientId]/qa` - Search approved medical records

### Direct FastAPI Endpoints (port 8000)
- `POST /auth/signup` - User registration
- `POST /auth/login` - Authentication with JWT
- `POST /summarize/{session_id}` - AI-powered summarization
- Authentication required for all medical data endpoints

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.8+ and pip (for backend components)
- Supabase account and project
- Git

### 1. Clone and Setup
```bash
git clone <repository-url>
cd nightingale-ai
```

### 2. Database Setup
1. Create a new Supabase project at https://supabase.com
2. Run the SQL scripts in the root folder to set up tables:
   - `001_create_tables.sql` - Creates profiles, sessions, segments, and summaries tables with RLS
   - `002_profile_trigger.sql` - Auto-creates profiles on user registration

### 3. Environment Configuration
Copy `.env.example` to `.env.local` and configure your environment variables:
```bash
cp .env.example .env.local
```

Update the following variables:
```env
# Frontend (required)
NEXT_PUBLIC_SUPABASE_URL=your-supabase-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000

# Backend (for AI processing)
DB_ENCRYPTION_KEY=your-encryption-key-here
SECRET_KEY=your-secret-key-here
MASTER_PASSWORD=your-master-password-here
```

### 4. Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: http://localhost:3000

### 5. Backend Setup (Optional - for AI processing)
```bash
# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server
python scripts/start_secure_backend.py
```

Backend will be available at: http://127.0.0.1:8000
API Documentation: http://127.0.0.1:8000/docs

##  Development


### Project Structure
```
nightingale-ai/
├── app/                   # Next.js App Router pages
│   ├── auth/             # Authentication pages (login, signup, callback)
│   ├── patient/          # Patient dashboard and consent management
│   ├── doctor/           # Doctor dashboard with dynamic user names
│   ├── session/          # Session management and markdown summaries
│   ├── api/              # API routes with comprehensive endpoints
│   ├── layout.tsx        # Root layout with Supabase integration
│   ├── page.tsx          # Landing page with medical features
│   └── globals.css       # Global styles with Tailwind CSS v4
├── components/           # Reusable UI components
│   ├── ui/              # shadcn/ui components
│   ├── auth/            # Authentication forms and components
│   └── custom/          # Custom application components
├── lib/                 # Utilities and configurations
│   ├── supabase/        # Supabase client, server, and middleware
│   ├── auth.ts          # Authentication utilities
│   └── utils.ts         # Utility functions
├── hooks/               # Custom React hooks
├── scripts/             # Backend utilities
│   └── start_secure_backend.py  # Secure backend startup script
├── summary_training/    # Local AI model training infrastructure
├── 001_create_tables.sql       # Database schema with RLS policies
├── 002_profile_trigger.sql     # Auto-profile creation trigger
├── backend/             # FastAPI Backend
│   ├── app.py          # FastAPI main application with all endpoints
│   ├── models.py       # SQLAlchemy database models
│   ├── security.py     # JWT authentication and password hashing
│   ├── redact.py       # PHI redaction and privacy protection
│   ├── summarize.py    # AI summarization with dual outputs
│   └── tests/          # Comprehensive test suite
└── requirements.txt    # Python dependencies including AI libraries
```


### Running Tests
```bash
# Backend tests
cd backend
python -m pytest tests/ -v

# Frontend tests (if implemented)
npm test
```

##  Deployment

### Vercel Deployment (Recommended)
1. Connect your GitHub repository to Vercel
2. Configure environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Manual Deployment
```bash
# Build the application
npm run build

# Deploy to your preferred platform
# Ensure all environment variables are configured
```

##  Database Schema

### Core Tables
- **profiles**: User information with role-based access (patient/doctor)
- **sessions**: Medical consultation sessions with status tracking
- **segments**: Audio transcription segments with PHI redaction
- **summaries**: AI-generated session summaries with provenance tracking
- **auth.users**: Supabase authentication (managed)

### Security Features
- **Row Level Security (RLS)**: Comprehensive policies for all tables
- **Role-based Access Control**: Separate permissions for patients and doctors
- **Secure Session Management**: Session isolation and access control
- **Audit Trail**: Complete logging for all medical data access

### RLS Policies Implementation
- Users can only view/modify their own profiles
- Session access limited to patient and assigned doctor
- Segments accessible only to session participants
- Summaries restricted to authorized session participants
- Cross-table relationship validation for data integrity

## 📄 License & Attribution

See `Attribution.txt` for detailed licensing information.

This project uses modern web technologies and follows healthcare industry best practices for privacy and security.

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request
5. Ensure all tests pass

##  Support

For questions or support:
- Open an issue in the repository
- Review the documentation
- Check the test suite for examples

---

**Built with privacy, security, and medical accuracy at its core.**
