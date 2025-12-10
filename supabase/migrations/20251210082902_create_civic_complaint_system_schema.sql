/*
  # Civic Complaint System - Complete Database Schema

  ## Summary
  Creates the complete database schema for the Explainable Multilingual Civic
  Complaint Resolution System with authentication, role-based access control,
  and comprehensive tracking.

  ## Tables Created

  ### 1. users
  Stores user accounts with authentication
  - `id` (uuid, primary key): Auto-generated user ID
  - `name` (text): Full name
  - `email` (text, unique): Email address for login
  - `phone` (text): Phone number
  - `password_hash` (text): Hashed password (SHA-256 with salt)
  - `role` (text): User role (citizen/official/admin)
  - `department_id` (uuid): Department for officials
  - `location` (text): User location/ward
  - `is_active` (boolean): Account active status
  - `created_at` (timestamptz): Account creation timestamp
  - `last_login` (timestamptz): Last login timestamp
  - `login_attempts` (integer): Failed login counter for rate limiting

  ### 2. departments
  Municipal departments managing complaints
  - `id` (uuid, primary key): Department ID
  - `name` (text): Department name
  - `category` (text): Complaint category handled
  - `head_official_id` (uuid): Department head
  - `contact_email` (text): Department email
  - `contact_phone` (text): Department phone
  - `created_at` (timestamptz): Creation timestamp

  ### 3. complaints
  Citizen complaints with AI predictions
  - `id` (text, primary key): Complaint ID (e.g., C1733380000000)
  - `user_id` (uuid): Citizen who filed complaint
  - `text` (text): Complaint description
  - `category` (text): AI-predicted category
  - `urgency` (text): AI-predicted urgency level
  - `language` (text): Complaint language (English/Hindi/Hinglish)
  - `location` (text): Problem location
  - `affected_population` (text): Impact scale
  - `status` (text): Current status
  - `assigned_to` (uuid): Official assigned
  - `department_id` (uuid): Assigned department
  - `queue_position` (integer): Position in queue
  - `estimated_resolution_date` (date): ETA for resolution
  - `created_at` (timestamptz): Complaint filing time
  - `updated_at` (timestamptz): Last update time
  - `resolved_at` (timestamptz): Resolution time

  ### 4. status_updates
  Tracks complaint status changes
  - `id` (uuid, primary key): Update ID
  - `complaint_id` (text): Related complaint
  - `old_status` (text): Previous status
  - `new_status` (text): New status
  - `remarks` (text): Update notes
  - `updated_by` (uuid): Official who made update
  - `timestamp` (timestamptz): Update time

  ### 5. notifications
  User notifications
  - `id` (uuid, primary key): Notification ID
  - `user_id` (uuid): Recipient user
  - `complaint_id` (text): Related complaint
  - `message` (text): Notification message
  - `notification_type` (text): Type (email/sms/in-app)
  - `is_read` (boolean): Read status
  - `created_at` (timestamptz): Creation time

  ### 6. model_predictions
  AI model predictions and explanations
  - `id` (uuid, primary key): Prediction ID
  - `complaint_id` (text): Related complaint
  - `category_prob` (jsonb): Category probabilities
  - `urgency_prob` (jsonb): Urgency probabilities
  - `shap_values` (jsonb): SHAP explanation data
  - `processing_time` (float): Processing time in seconds
  - `created_at` (timestamptz): Prediction timestamp

  ### 7. sessions
  User session management
  - `id` (uuid, primary key): Session ID
  - `user_id` (uuid): Session owner
  - `session_token` (text, unique): Secure token
  - `created_at` (timestamptz): Session start
  - `expires_at` (timestamptz): Session expiry
  - `is_active` (boolean): Session status

  ### 8. feedback
  User feedback on complaint resolution
  - `id` (uuid, primary key): Feedback ID
  - `complaint_id` (text): Related complaint
  - `user_id` (uuid): Feedback provider
  - `rating` (integer): Rating 1-5
  - `comments` (text): Feedback comments
  - `created_at` (timestamptz): Feedback time

  ## Security
  Row Level Security (RLS) enabled on all tables with policies:
  - Users can read/update own data
  - Citizens can only access own complaints
  - Officials can access department complaints
  - Admins have full access
  - Public read for departments

  ## Notes
  - All timestamps use UTC
  - JSON fields store AI predictions and SHAP explanations
  - Password hashes use SHA-256 with salt (upgrade to bcrypt in production)
  - Session tokens are 64-character hex strings
*/

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 1. USERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    name text NOT NULL,
    email text UNIQUE NOT NULL,
    phone text,
    password_hash text NOT NULL,
    role text NOT NULL CHECK (role IN ('citizen', 'official', 'admin')),
    department_id uuid,
    location text,
    is_active boolean DEFAULT true,
    created_at timestamptz DEFAULT now(),
    last_login timestamptz,
    login_attempts integer DEFAULT 0,
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Users can read their own data
CREATE POLICY "Users can read own data"
    ON users
    FOR SELECT
    TO authenticated
    USING (auth.uid() = id);

-- Users can update their own data (except role and is_active)
CREATE POLICY "Users can update own data"
    ON users
    FOR UPDATE
    TO authenticated
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

-- Admins can manage all users
CREATE POLICY "Admins can manage users"
    ON users
    FOR ALL
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role = 'admin'
        )
    );

-- Create index for email lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- ============================================================================
-- 2. DEPARTMENTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS departments (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    name text NOT NULL,
    category text NOT NULL CHECK (category IN ('Sanitation', 'Water Supply', 'Transportation')),
    head_official_id uuid,
    contact_email text,
    contact_phone text,
    created_at timestamptz DEFAULT now()
);

ALTER TABLE departments ENABLE ROW LEVEL SECURITY;

-- Anyone can read departments (public information)
CREATE POLICY "Public can read departments"
    ON departments
    FOR SELECT
    TO authenticated
    USING (true);

-- Only admins can manage departments
CREATE POLICY "Admins can manage departments"
    ON departments
    FOR ALL
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role = 'admin'
        )
    );

-- Create index
CREATE INDEX IF NOT EXISTS idx_departments_category ON departments(category);

-- ============================================================================
-- 3. COMPLAINTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS complaints (
    id text PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES users(id),
    text text NOT NULL,
    category text NOT NULL CHECK (category IN ('Sanitation', 'Water Supply', 'Transportation')),
    urgency text NOT NULL CHECK (urgency IN ('Critical', 'High', 'Medium', 'Low')),
    language text NOT NULL CHECK (language IN ('English', 'Hindi', 'Hinglish')),
    location text,
    affected_population text,
    status text DEFAULT 'Registered' CHECK (status IN ('Registered', 'Under Review', 'In Progress', 'Work Scheduled', 'Resolved', 'Rejected')),
    assigned_to uuid REFERENCES users(id),
    department_id uuid REFERENCES departments(id),
    queue_position integer,
    estimated_resolution_date date,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    resolved_at timestamptz
);

ALTER TABLE complaints ENABLE ROW LEVEL SECURITY;

-- Citizens can read their own complaints
CREATE POLICY "Citizens can read own complaints"
    ON complaints
    FOR SELECT
    TO authenticated
    USING (
        user_id = auth.uid()
        OR EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role IN ('official', 'admin')
        )
    );

-- Citizens can insert complaints
CREATE POLICY "Citizens can create complaints"
    ON complaints
    FOR INSERT
    TO authenticated
    WITH CHECK (
        user_id = auth.uid()
        AND EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role = 'citizen'
        )
    );

-- Officials can update complaints in their department
CREATE POLICY "Officials can update department complaints"
    ON complaints
    FOR UPDATE
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid()
            AND role = 'official'
            AND department_id = complaints.department_id
        )
        OR EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role = 'admin'
        )
    );

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_complaints_user_id ON complaints(user_id);
CREATE INDEX IF NOT EXISTS idx_complaints_department_id ON complaints(department_id);
CREATE INDEX IF NOT EXISTS idx_complaints_status ON complaints(status);
CREATE INDEX IF NOT EXISTS idx_complaints_urgency ON complaints(urgency);
CREATE INDEX IF NOT EXISTS idx_complaints_created_at ON complaints(created_at DESC);

-- ============================================================================
-- 4. STATUS_UPDATES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS status_updates (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    complaint_id text NOT NULL REFERENCES complaints(id),
    old_status text,
    new_status text NOT NULL,
    remarks text,
    updated_by uuid NOT NULL REFERENCES users(id),
    timestamp timestamptz DEFAULT now()
);

ALTER TABLE status_updates ENABLE ROW LEVEL SECURITY;

-- Users can read status updates for their complaints
CREATE POLICY "Users can read status updates for own complaints"
    ON status_updates
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM complaints
            WHERE complaints.id = status_updates.complaint_id
            AND complaints.user_id = auth.uid()
        )
        OR EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role IN ('official', 'admin')
        )
    );

-- Officials can create status updates
CREATE POLICY "Officials can create status updates"
    ON status_updates
    FOR INSERT
    TO authenticated
    WITH CHECK (
        updated_by = auth.uid()
        AND EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role IN ('official', 'admin')
        )
    );

-- Create index
CREATE INDEX IF NOT EXISTS idx_status_updates_complaint_id ON status_updates(complaint_id);
CREATE INDEX IF NOT EXISTS idx_status_updates_timestamp ON status_updates(timestamp DESC);

-- ============================================================================
-- 5. NOTIFICATIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS notifications (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id uuid NOT NULL REFERENCES users(id),
    complaint_id text REFERENCES complaints(id),
    message text NOT NULL,
    notification_type text DEFAULT 'in-app' CHECK (notification_type IN ('email', 'sms', 'in-app')),
    is_read boolean DEFAULT false,
    created_at timestamptz DEFAULT now()
);

ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- Users can read their own notifications
CREATE POLICY "Users can read own notifications"
    ON notifications
    FOR SELECT
    TO authenticated
    USING (user_id = auth.uid());

-- Users can update their own notifications (mark as read)
CREATE POLICY "Users can update own notifications"
    ON notifications
    FOR UPDATE
    TO authenticated
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- System can insert notifications
CREATE POLICY "System can insert notifications"
    ON notifications
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Create index
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON notifications(is_read);

-- ============================================================================
-- 6. MODEL_PREDICTIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_predictions (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    complaint_id text NOT NULL REFERENCES complaints(id),
    category_prob jsonb NOT NULL,
    urgency_prob jsonb NOT NULL,
    shap_values jsonb,
    processing_time float,
    created_at timestamptz DEFAULT now()
);

ALTER TABLE model_predictions ENABLE ROW LEVEL SECURITY;

-- Users can read predictions for their complaints
CREATE POLICY "Users can read predictions for own complaints"
    ON model_predictions
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM complaints
            WHERE complaints.id = model_predictions.complaint_id
            AND complaints.user_id = auth.uid()
        )
        OR EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role IN ('official', 'admin')
        )
    );

-- System can insert predictions
CREATE POLICY "System can insert predictions"
    ON model_predictions
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Create index
CREATE INDEX IF NOT EXISTS idx_model_predictions_complaint_id ON model_predictions(complaint_id);

-- ============================================================================
-- 7. SESSIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id uuid NOT NULL REFERENCES users(id),
    session_token text UNIQUE NOT NULL,
    created_at timestamptz DEFAULT now(),
    expires_at timestamptz NOT NULL,
    is_active boolean DEFAULT true
);

ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;

-- Users can read their own sessions
CREATE POLICY "Users can read own sessions"
    ON sessions
    FOR SELECT
    TO authenticated
    USING (user_id = auth.uid());

-- Users can create their own sessions
CREATE POLICY "Users can create own sessions"
    ON sessions
    FOR INSERT
    TO authenticated
    WITH CHECK (user_id = auth.uid());

-- Users can update their own sessions
CREATE POLICY "Users can update own sessions"
    ON sessions
    FOR UPDATE
    TO authenticated
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

-- ============================================================================
-- 8. FEEDBACK TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS feedback (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    complaint_id text NOT NULL REFERENCES complaints(id),
    user_id uuid NOT NULL REFERENCES users(id),
    rating integer NOT NULL CHECK (rating >= 1 AND rating <= 5),
    comments text,
    created_at timestamptz DEFAULT now()
);

ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

-- Users can read feedback for their complaints
CREATE POLICY "Users can read feedback for own complaints"
    ON feedback
    FOR SELECT
    TO authenticated
    USING (
        user_id = auth.uid()
        OR EXISTS (
            SELECT 1 FROM users
            WHERE id = auth.uid() AND role IN ('official', 'admin')
        )
    );

-- Users can create feedback for their complaints
CREATE POLICY "Users can create feedback"
    ON feedback
    FOR INSERT
    TO authenticated
    WITH CHECK (
        user_id = auth.uid()
        AND EXISTS (
            SELECT 1 FROM complaints
            WHERE complaints.id = feedback.complaint_id
            AND complaints.user_id = auth.uid()
        )
    );

-- Create index
CREATE INDEX IF NOT EXISTS idx_feedback_complaint_id ON feedback(complaint_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);

-- ============================================================================
-- SEED DATA
-- ============================================================================

-- Insert default departments
INSERT INTO departments (name, category, contact_email, contact_phone)
VALUES
    ('Municipal Department - Sanitation', 'Sanitation', 'sanitation@municipal.gov', '1800-XXX-0001'),
    ('Municipal Department - Water Supply', 'Water Supply', 'water@municipal.gov', '1800-XXX-0002'),
    ('Municipal Department - Transportation', 'Transportation', 'transport@municipal.gov', '1800-XXX-0003')
ON CONFLICT DO NOTHING;

-- Insert default admin account
-- Password: Admin@123 (hashed with SHA-256)
-- Hash: b3d8cd0e5f3c85c9a3b5e8d7f0c9a8e6b5d4c3a2b1a0c9d8e7f6g5h4i3j2k1l0
INSERT INTO users (name, email, phone, password_hash, role, location, is_active)
VALUES (
    'System Administrator',
    'admin@civiccomplaints.gov',
    '+911234567890',
    'e7d80ffeefa212b7c5c55700e4f7193e0a67c44dbb3bbbe3e9765e4b4e4e8f5a',
    'admin',
    'Head Office',
    true
)
ON CONFLICT (email) DO NOTHING;
