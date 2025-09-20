# Nightingale AI - Medical Voice Assistant

A comprehensive medical AI system featuring privacy-first voice consultations with a **Next.js frontend** and **FastAPI backend** that demonstrates a **safe, provenance-grounded VoiceAI consultation workflow**.

##  Architecture

- **Frontend**: Next.js 15 with TypeScript, Tailwind CSS v4, and shadcn/ui components
- **Backend**: FastAPI with SQLAlchemy, JWT authentication, and privacy-focused AI processing  
- **Database**: Supabase (PostgreSQL) with Row Level Security
- **Authentication**: Supabase Auth with role-based access control

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

### Authentication Endpoints
- `POST /api/auth/login` - User authentication with role-based redirection
- `POST /api/auth/signup` - User registration with role selection

### Session Management
- `POST /api/session/start` - Initialize new medical consultation session
- `GET /api/session/[id]/summary` - Retrieve session summary with markdown rendering

### Medical Data Processing
- `POST /api/asr/ingest` - Ingest automatic speech recognition segments
- `POST /api/summarize/[sessionId]` - Generate AI-powered medical summaries
- `GET /api/summary/[summaryId]` - Retrieve specific medical summary

### Review & Approval Workflow
- `POST /api/review/[summaryId]/approve` - Doctor approval of patient summaries
- `POST /api/consent/record` - Record patient consent for AI processing

### Patient Services
- `GET /api/patient/[patientId]/qa` - Search patient's approved medical records

##  Security Implementation

### Database Security
- **Row Level Security (RLS)**: Comprehensive policies for all tables
  - Users can only access their own profiles
  - Patients and assigned doctors can access session data
  - Cross-table relationship validation
- **Secure Relationships**: Foreign key constraints with cascade policies
- **Audit Logging**: Automatic tracking of all data modifications

### Authentication Security
- **Supabase Auth Integration**: Secure authentication with automatic profile creation
- **Role Validation**: Server-side role verification for all protected routes
- **Session Security**: Automatic token refresh and cleanup
- **Dynamic User Display**: Real-time fetching of user information for personalized interfaces

### API Security
- **Bearer Token Authentication**: Supabase JWT token validation
- **Input Validation**: Comprehensive request validation and sanitization
- **Error Handling**: Secure error responses without information leakage
- **CORS Configuration**: Proper cross-origin resource sharing setup

### Privacy Protection
- **PHI Detection**: Automatic identification of sensitive medical information
- **Data Redaction**: Real-time redaction before AI processing
- **Consent Tracking**: Explicit consent management with version control
- **Access Logging**: Complete audit trail for compliance

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
2. Run the SQL scripts in the `scripts/` folder to set up tables:
   - `001_create_tables.sql` - Creates profiles, sessions, segments, and summaries tables with RLS
   - `002_profile_trigger.sql` - Auto-creates profiles on user registration

### 3. Environment Configuration
Set up your environment variables in Vercel or your deployment platform:
```env
SUPABASE_URL=your-supabase-project-url
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
NEXT_PUBLIC_SUPABASE_URL=your-supabase-project-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
NEXT_PUBLIC_DEV_SUPABASE_REDIRECT_URL=http://localhost:3000
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
python scripts/start_backend.py
```

Backend will be available at: http://127.0.0.1:8000
API Documentation: http://127.0.0.1:8000/docs

##  Complete Workflow

### Pre-Care
- **Authentication**: Secure Supabase Auth with email/password
- **Role Selection**: Separate registration for patients and healthcare providers
- **Consent Management**: Explicit consent recording before any AI processing
- **Profile Management**: User profiles with role-based access control

### During Care
- **Voice Recording**: Real-time audio capture and transcription
- **Privacy Protection**: Automatic PHI redaction before AI processing
- **Session Management**: Secure session tracking with unique identifiers
- **Real-time Processing**: Live conversation analysis and note-taking

### Post-Care
- **AI Summarization**: Generate both clinician and patient-friendly summaries
- **Provenance Tracking**: Every summary point links back to original conversation
- **Human Review**: Doctors can review and approve summaries
- **Patient Access**: Secure patient portal for accessing approved summaries

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
├── scripts/             # Database scripts and utilities
│   ├── 001_create_tables.sql    # Database schema with RLS policies
│   ├── 002_profile_trigger.sql  # Auto-profile creation trigger
│   └── start_backend.py         # Backend startup script
├── backend/             # FastAPI Backend
│   ├── app.py          # FastAPI main application with all endpoints
│   ├── models.py       # SQLAlchemy database models
│   ├── security.py     # JWT authentication and password hashing
│   ├── redact.py       # PHI redaction and privacy protection
│   ├── summarize.py    # AI summarization with dual outputs
│   └── tests/          # Comprehensive test suite
└── requirements.txt    # Python dependencies including AI libraries
```

##  Testing

### Quality Assurance Tests

The project includes comprehensive tests for medical AI safety:

#### 1. Grounding Validation
- Ensures every summary point has source attribution
- Validates traceability of AI-generated content
- Critical for medical accuracy and accountability

#### 2. Privacy Protection
- Tests PHI redaction on synthetic medical data
- Validates HIPAA compliance measures
- Ensures no sensitive information leaks

#### 3. Performance Profiling
- Monitors system latency and response times
- Tests scalability with varying loads
- Ensures real-time performance requirements

#### 4. Summary Quality
- Compares clinician vs patient summary formats
- Tests appropriate medical terminology usage
- Validates user-centric design choices

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
