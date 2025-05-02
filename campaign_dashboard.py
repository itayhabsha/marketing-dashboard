import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import base64
import streamlit.components.v1 as components



# ====== Page Config ======
st.set_page_config(
    page_title="Marketing Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== Force Scroll to Top ======
st.markdown("""
    <script>
        window.scrollTo(0, 0);
    </script>
""", unsafe_allow_html=True)


# ====== Sidebar Logo ======
with open("ResonLabs.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

with st.sidebar:
    st.markdown(f"""
        <div style='
            text-align: center;
            padding-top: 0px;
            padding-bottom: 0px;
            margin-bottom: 2px;
        '>
            <img src="data:image/png;base64,{encoded}" style='width: 240px;' />
        </div>
    """, unsafe_allow_html=True)


# ====== Apply Global CSS ======
st.markdown("""
<style>
/* ========== GLOBAL STYLES ========== */
/* Colors */
:root {
    --primary-color: #4361EE;
    --secondary-color: #00C48C;
    --accent-color-1: #F72585;
    --accent-color-2: #8C54FF;
    --accent-color-3: #3A0CA3;
    --accent-color-4: #4CC9F0;
    --dark-color: #2c3e50;
    --light-color: #f8f9fe;
    --text-primary: #2c3e50;
    --text-secondary: #64748b;
    --text-muted: #7f8fa4;
    --text-light: #a3aed0;
    --border-color: #e2e8f0;
    --box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
}

/* Text Styles */
.dashboard-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    font-family: 'Segoe UI', Arial, sans-serif;
    text-align: left;
}

.dashboard-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-bottom: 2rem;
    font-family: 'Segoe UI', Arial, sans-serif;
    text-align: left;
}

.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
    font-family: 'Segoe UI', Arial, sans-serif;
}

/* Container Styles */
.dashboard-container {
    background-color: var(--light-color);
    border-radius: 12px;
    padding: 20px 30px;
    margin-bottom: 2rem;
    box-shadow: var(--box-shadow);
}

.chart-container {
    background-color: white;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* Metrics Display */
.metric-container {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    flex: 1;
    min-width: 200px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    border-top: 4px solid var(--primary-color);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-label {
    font-size: 19px;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 1px;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 10px 0;
}

.metric-subtext {
    font-size: 18px;
    color: var(--text-light);
}

/* Card Color Variants */
.card-primary { border-top: 4px solid var(--primary-color); }
.card-secondary { border-top: 4px solid var(--secondary-color); }
.card-accent-1 { border-top: 4px solid var(--accent-color-1); }
.card-accent-2 { border-top: 4px solid var(--accent-color-2); }
.card-accent-3 { border-top: 4px solid var(--accent-color-3); }
.card-accent-4 { border-top: 4px solid var(--accent-color-4); }

/* Insights Box */
.insight-box {
    background-color: #ffffff; 
    border: 2px solid var(--text-primary); 
    border-radius: 12px; 
    padding: 20px; 
    text-align: center; 
    box-shadow: 0 2px 6px rgba(0,0,0,0.08); 
    margin-top: 24px;
}

.insight-title {
    font-size: 21px; 
    color: var(--text-primary); 
    font-weight: 600; 
    margin-bottom: 10px;
}

.insight-metric {
    font-size: 28px; 
    font-weight: 700; 
    color: #1a202c; 
    margin-bottom: 12px;
}

.insight-details {
    font-size: 21px; 
    color: #4a5568; 
    line-height: 1.6;
}

/* Sidebar Navigation */
.sidebar-header {
    font-size: 1.8rem;
    font-weight: 700;
    text-align: center;
    color: var(--text-primary);
    margin-bottom: 20px;
}

.nav-button {
    display: block;
    width: 100%;
    background-color: #f0f2f6;
    color: #333;
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 12px 0;
    text-align: center;
    margin-bottom: 10px;
    font-size: 16px;
    font-weight: 600;
    text-decoration: none;
    transition: background-color 0.2s ease;
}

.nav-button:hover {
    background-color: #e4e8f0;
    cursor: pointer;
}

/* ========== CUSTOM: DataFrame Font Size ========== */
div[data-testid="stDataFrame"] div[role="gridcell"] {{
    font-size: 20px !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}}
</style>
""", unsafe_allow_html=True)

# ====== Load Data ======
@st.cache_data
def load_data():
    return pd.read_excel("my_data.xlsx")

df = load_data()

# ====== Initialize Session State ======
if "page" not in st.session_state:
    st.session_state.page = "home"

# ====== Sidebar Navigation ======
with st.sidebar:
    st.markdown("<div class='sidebar-header'>Navigation</div>", unsafe_allow_html=True)

    with st.form("home_form"):
        submitted = st.form_submit_button("Home", use_container_width=True)
        if submitted:
            st.session_state.page = "home"
            

    with st.form("compare_form"):
        submitted = st.form_submit_button("Compare Campaigns", use_container_width=True)
        if submitted:
            st.session_state.page = "compare_campaigns"

    with st.form("analyze_form"):
        if "rerun_scroll" not in st.session_state:
            st.session_state.rerun_scroll = False
        submitted = st.form_submit_button("Analyze Single Campaign", use_container_width=True)
        if submitted:
            st.session_state.page = "single_campaign"

# ====== Data Preprocessing ======
df['finished_quiz'] = (df['safety_level_quiz_score'] > 0).astype(int)

# ====== Configure Matplotlib Default Styling ======
def set_plot_style(fig, ax):
    # Set common matplotlib styling
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('#DDE1E4')
    ax.spines['bottom'].set_color('#DDE1E4')
    
    ax.grid(axis='both', linestyle='--', alpha=0.15)
    ax.tick_params(axis='both', labelsize=8, colors="#333")
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_axisbelow(True)
    
    return fig, ax

# ====== ============Home Page ======================================================================================================================================================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="dashboard-container">
        <div class="dashboard-title">Marketing Insights Dashboard</div>
        <div class="dashboard-subtitle">Analyze campaign performance and optimize marketing strategies</div>
    </div>
    """, unsafe_allow_html=True)

    if not df.empty:
        total_campaigns = df['Campaign number'].nunique()
        total_users = df['ruserid'].nunique()
        total_purchases = df['purcheas_ind'].sum()
        conversion_rate = (total_purchases / total_users) * 100 if total_users > 0 else 0

        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card card-primary">
                <div class="metric-label">CAMPAIGNS</div>
                <div class="metric-value">{total_campaigns:,}</div>
                <div class="metric-subtext">Active Campaigns</div>
            </div>
            <div class="metric-card card-accent-2">
                <div class="metric-label">USERS</div>
                <div class="metric-value">{total_users:,}</div>
                <div class="metric-subtext">Unique Visitors</div>
            </div>
            <div class="metric-card card-secondary">
                <div class="metric-label">CONVERSION RATE</div>
                <div class="metric-value">{conversion_rate:.2f}%</div>
                <div class="metric-subtext">Overall Performance</div>
            </div>
            <div class="metric-card card-accent-1">
                <div class="metric-label">PURCHASES</div>
                <div class="metric-value">{int(total_purchases):,}</div>
                <div class="metric-subtext">Total Completed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Active Campaigns Summary</div>", unsafe_allow_html=True)

        campaign_summary = (
            df.groupby(['campaign', 'Campaign number'])
            .agg(Users=('ruserid', 'nunique'), Purchases=('purcheas_ind', 'sum'))
            .reset_index()
        )
        campaign_summary['Conversion Rate'] = (campaign_summary['Purchases'] / campaign_summary['Users']) * 100
        campaign_summary['Conversion Rate'] = campaign_summary['Conversion Rate'].round(2).astype(str) + '%'

        campaign_summary = campaign_summary.sort_values(by='Campaign number')

        st.data_editor(
            campaign_summary,
            use_container_width=True,
            hide_index=True,
            disabled=True
        )

# ====== Compare Campaigns Page ==============================================================================================================================================================================
elif st.session_state.page == "compare_campaigns":
    st.markdown("""
    <div class="dashboard-container">
        <div class="dashboard-title">Compare Campaigns</div>
        <div class="dashboard-subtitle">Compare two campaigns and analyze performance uplift</div>
    </div>
    """, unsafe_allow_html=True)

    if not df.empty:
        # Prepare campaign data
        campaign_df = (
            df[['campaign', 'Campaign number']]
            .drop_duplicates()
            .dropna()
            .rename(columns={'campaign': 'Campaign Name', 'Campaign number': 'Campaign Number'})
            .reset_index(drop=True)
        )
        campaign_df["Campaign Number"] = campaign_df["Campaign Number"].astype(int)
        camp_options = sorted(campaign_df["Campaign Number"].unique())
        camp_options = [str(c) for c in camp_options]

        # Campaign selection
        col_select = st.columns(2)
        with col_select[0]:
            camp_A_number = st.selectbox("Select Campaign A", camp_options)
        with col_select[1]:
            camp_B_number = st.selectbox("Select Campaign B", [c for c in camp_options if c != camp_A_number])

        # Get campaign names and filter data
        camp_A_name = campaign_df[campaign_df["Campaign Number"].astype(str) == camp_A_number]["Campaign Name"].values[0]
        camp_B_name = campaign_df[campaign_df["Campaign Number"].astype(str) == camp_B_number]["Campaign Name"].values[0]

        df_A = df[df['campaign'] == camp_A_name]
        df_B = df[df['campaign'] == camp_B_name]

        # Calculate metrics
        total_users_A = df_A['ruserid'].nunique()
        total_purchases_A = df_A['purcheas_ind'].sum()
        total_users_B = df_B['ruserid'].nunique()
        total_purchases_B = df_B['purcheas_ind'].sum()

        conversion_A = (total_purchases_A / total_users_A) * 100 if total_users_A > 0 else 0
        conversion_B = (total_purchases_B / total_users_B) * 100 if total_users_B > 0 else 0
        uplift = conversion_B - conversion_A

        # Calculate Bayesian probability
        samples_A = np.random.beta(total_purchases_A + 1, total_users_A - total_purchases_A + 1, 5000)
        samples_B = np.random.beta(total_purchases_B + 1, total_users_B - total_purchases_B + 1, 5000)
        prob_B_better = np.mean(samples_B > samples_A) * 100

        # Display metrics
        st.markdown("<div class='section-title'>Campaign Performance Summary</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card card-primary">
                <div class="metric-label">Campaign {camp_A_number} Conversion Rate</div>
                <div class="metric-value">{conversion_A:.2f}%</div>
                <div class="metric-subtext">{total_users_A:,} Users | {total_purchases_A:,} Purchases</div>
            </div>
            <div class="metric-card card-secondary">
                <div class="metric-label">Campaign {camp_B_number} Conversion Rate</div>
                <div class="metric-value">{conversion_B:.2f}%</div>
                <div class="metric-subtext">{total_users_B:,} Users | {total_purchases_B:,} Purchases</div>
            </div>
            <div class="metric-card card-accent-1">
                <div class="metric-label">Conversion Uplift</div>
                <div class="metric-value">{uplift:+.2f}%</div>
                <div class="metric-subtext">Campaign {camp_B_number} vs Campaign {camp_A_number}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Layout for charts and recommendation
        layout_cols = st.columns(2)

        # Left column: Probability distributions
        with layout_cols[0]:
            st.markdown("<div class='section-title'>Probability Distributions</div>", unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(7, 4.5))
            
            x = np.linspace(0, max(samples_A.max(), samples_B.max()) * 1.1, 1000)
            ax.plot(x, beta.pdf(x, total_purchases_A + 1, total_users_A - total_purchases_A + 1),
                    label=f"Campaign {camp_A_number}", color="#4F46E5", linewidth=2)
            ax.plot(x, beta.pdf(x, total_purchases_B + 1, total_users_B - total_purchases_B + 1),
                    label=f"Campaign {camp_B_number}", color="#00BFFF", linewidth=2)
            
            ax.set_xlabel('Conversion Rate')
            ax.set_ylabel('Density')
            ax.set_title('Posterior Probability Distributions')
            ax.legend(frameon=False)
            
            fig, ax = set_plot_style(fig, ax)
            plt.tight_layout()
            st.pyplot(fig)

        # Right column: Campaign recommendation
        with layout_cols[1]:
            winner = f"Campaign {camp_B_number}" if prob_B_better > 50 else f"Campaign {camp_A_number}"
            certainty = max(prob_B_better, 100 - prob_B_better)

            st.markdown("<div class='section-title'>Campaign Recommendation</div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">Campaign Performance Recommendation</div>
                    <div class="insight-metric">{winner}</div>
                    <div class="insight-details">Campaign <strong>{winner}</strong> has a <strong>{certainty:.2f}%</strong> probability of achieving better conversion results.</div>
                </div>
            """, unsafe_allow_html=True)

# ====== Analyze Single Campaign Page ======================================================================================================================================================================================
elif st.session_state.page == "single_campaign":
    components.html(
        """
        <script>
            window.parent.scrollTo(0, 0);
        </script>
        """,
        height=0
    )

    with st.container():
        st.markdown("""
<div class="dashboard-container">
    <div class="dashboard-title">Campaign Analysis</div>
    <div class="dashboard-subtitle">Deep dive into performance metrics and user behavior</div>
</div>
""", unsafe_allow_html=True)


        if not df.empty:
            # ====== Select Campaign ======
            st.markdown("<div class='section-title'>Select Campaign</div>", unsafe_allow_html=True)

            campaign_df = (
                df[['campaign', 'Campaign number']]
                .drop_duplicates()
                .dropna()
                .rename(columns={'campaign': 'Campaign Name', 'Campaign number': 'Campaign Number'})
                .reset_index(drop=True)
            )
            campaign_df['Campaign Number'] = pd.to_numeric(campaign_df['Campaign Number'], errors='coerce')
            campaign_df['Campaign Number'] = campaign_df['Campaign Number'].astype(int)
            campaign_df = campaign_df.sort_values('Campaign Number').reset_index(drop=True)
            camp_options = campaign_df["Campaign Number"].astype(str).tolist()

            col1, col2 = st.columns([3, 1])
            with col1:
                selected_camp_number = st.selectbox("", camp_options, label_visibility="collapsed")

            selected_camp_name = campaign_df[campaign_df["Campaign Number"].astype(str) == selected_camp_number]["Campaign Name"].values[0]
            df_camp = df[df['campaign'] == selected_camp_name]

            # ====== Calculate Metrics ======
            total_users = df_camp['ruserid'].nunique()
            finished_quiz = df_camp['finished_quiz'].sum()
            transaction_start = df_camp['transaction_start'].sum() if 'transaction_start' in df_camp.columns else 0
            purchases = df_camp['purcheas_ind'].sum()
            conversion_rate = (purchases / total_users) * 100 if total_users > 0 else 0

            # ====== Display Key Metrics ======
            st.markdown("<div class='section-title'>Key Performance Metrics</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-card card-accent-3">
                    <div class="metric-label">Finished Quiz</div>
                    <div class="metric-value">{int(finished_quiz):,}</div>
                    <div class="metric-subtext">Completed Safety Quiz</div>
                </div>
                <div class="metric-card card-accent-4">
                    <div class="metric-label">Started Transaction</div>
                    <div class="metric-value">{int(transaction_start):,}</div>
                    <div class="metric-subtext">Checkout Started</div>
                </div>
                <div class="metric-card card-accent-1">
                    <div class="metric-label">Purchases</div>
                    <div class="metric-value">{int(purchases):,}</div>
                    <div class="metric-subtext">Successful Purchases</div>
                </div>
                <div class="metric-card card-secondary">
                    <div class="metric-label">Conversion Rate</div>
                    <div class="metric-value">{conversion_rate:.2f}%</div>
                    <div class="metric-subtext">Visitor-to-Buyer Rate</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            #====== User Journey Analysis ======
            # Section titles
            col_titles = st.columns([3, 2])
            with col_titles[0]:
                st.markdown("<div class='section-title'>User Journey Analysis</div>", unsafe_allow_html=True)
            with col_titles[1]:
                st.markdown("<div class='section-title'>Journey Insights</div>", unsafe_allow_html=True)

            # Main content: Graph + Summary Box
            journey_col1, journey_col2 = st.columns([3, 2])

            # Left column: Funnel graph
            with journey_col1:
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

                funnel_labels = ["Visitors", "Finished\nQuiz", "Started\nTransaction", "Purchases"]
                funnel_values = [total_users, int(finished_quiz), int(transaction_start), int(purchases)]

                fig, ax = plt.subplots(figsize=(7, 4.0))  # גובה מותאם
                colors = plt.cm.Blues(np.linspace(0.6, 0.95, len(funnel_labels)))

                bar_container = ax.barh(
                    funnel_labels,
                    funnel_values,
                    color=colors,
                    height=0.8  # יותר עבה – פחות רווחים
                )

                ax.set_xlabel("Users", fontsize=10, color="#555", labelpad=4)
                fig, ax = set_plot_style(fig, ax)

                for i, v in enumerate(funnel_values):
                    ax.text(v + max(funnel_values) * 0.01, i, f"{int(v):,}", va="center", fontsize=9, color="#555")

                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("</div>", unsafe_allow_html=True)


            # Right column: Insight box
            with journey_col2:
                quiz_rate = (finished_quiz / total_users * 100) if total_users > 0 else 0
                purchase_rate = (purchases / transaction_start * 100) if transaction_start > 0 else 0
                main_dropoff = 100 - quiz_rate if total_users > 0 else 0

                st.markdown(f"""
                    <div class="insight-box">
                        <div class="insight-title">User Journey Overview</div>
                        <div class="insight-metric">{quiz_rate:.1f}% Quiz Completion</div>
                        <div class="insight-details">
                            Purchase Rate After Transaction Start: <strong>{purchase_rate:.1f}%</strong><br>
                            Main Drop-off Before Quiz: <strong>{main_dropoff:.1f}%</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # ====== Feature Importance Analysis ======
            st.markdown("<div class='section-title'>Feature Importance Analysis</div>", unsafe_allow_html=True)

            # טאבים בלי כותרת insights
            tabs = st.tabs(["General Features", "Specific Answers"])


            # GENERAL FEATURES TAB
            with tabs[0]:
                insight_text = ""
                features_list = [
                    "use_the_internet_for_answered", "do_on_social_media_answered", 
                    "enter_personal_details_online_answered", "keep_your_passwords_answered", 
                    "victim_of_online_scam_answered", "nline_accounts_hacked_answered",
                    "safety_level_quiz_score", "breach_found"
                ]

                model_df = df_camp[features_list + ["purcheas_ind"]].dropna()

                if model_df.shape[0] > 30:
                    col_main = st.columns([3, 2])
                    with col_main[0]:
                        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

                        X = model_df.drop("purcheas_ind", axis=1)
                        y = model_df["purcheas_ind"]

                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X, y)

                        importances = pd.DataFrame({
                            "Feature": X.columns,
                            "Importance": model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)

                        feature_display_names = {
                            "use_the_internet_for_answered": "Internet Usage",
                            "do_on_social_media_answered": "Social Media Activity",
                            "enter_personal_details_online_answered": "Personal Details Sharing",
                            "keep_your_passwords_answered": "Password Management",
                            "victim_of_online_scam_answered": "Past Scam Victim",
                            "nline_accounts_hacked_answered": "Account Hacking History",
                            "safety_level_quiz_score": "Safety Quiz Score",
                            "breach_found": "Security Breach"
                        }

                        importances["Display"] = importances["Feature"].map(feature_display_names)

                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        colors = plt.cm.Blues(np.linspace(0.6, 0.95, len(importances)))
                        ax.barh(importances["Display"], importances["Importance"], color=colors)

                        ax.set_xlabel("Importance Score", fontsize=10, color="#555")
                        fig, ax = set_plot_style(fig, ax)

                        for i, v in enumerate(importances["Importance"]):
                            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color="#555")

                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)

                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_main[1]:
                        top_feature = importances.iloc[0]["Display"]
                        st.markdown(f"""
                            <div class="insight-box">
                                <div class="insight-title">Top Feature Insight</div>
                                <div class="insight-details">
                                    <strong>Strongest Predictor:</strong><br>
                                    {top_feature} is the strongest indicator for purchase.
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Not enough data for General Features analysis. Minimum 30 records required.")

            # SPECIFIC ANSWERS TAB
            with tabs[1]:
                insight_text = ""
                prefixes = [
                    "use_the_internet_for_", "do_on_social_media_", 
                    "enter_personal_details_online_", "keep_your_passwords_", 
                    "victim_of_online_scam_", "nline_accounts_hacked_"
                ]

                option_columns = [col for col in df.columns if any(col.startswith(p) for p in prefixes)]
                model_df_detail = df_camp[option_columns + ["purcheas_ind"]].dropna()

                if model_df_detail.shape[0] > 30:
                    col_main = st.columns([3, 2])
                    with col_main[0]:
                        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

                        X_detail = model_df_detail.drop("purcheas_ind", axis=1)
                        y_detail = model_df_detail["purcheas_ind"]

                        model_detail = RandomForestClassifier(n_estimators=100, random_state=42)
                        model_detail.fit(X_detail, y_detail)

                        importances_detail = pd.DataFrame({
                            "Feature": X_detail.columns,
                            "Importance": model_detail.feature_importances_
                        }).sort_values(by="Importance", ascending=False).head(10)

                        answer_mapping = {
                            "use_the_internet_for_1": "Social media",
                            "use_the_internet_for_2": "Banking & Finance",
                            "use_the_internet_for_3": "Online shopping",
                            "use_the_internet_for_4": "Gaming",
                            "use_the_internet_for_5": "Streaming",
                            "use_the_internet_for_6": "Research & Education",
                            "do_on_social_media_1": "News/Events",
                            "do_on_social_media_2": "Post Photos",
                            "do_on_social_media_3": "Entertainment",
                            "do_on_social_media_4": "Brand Research",
                            "enter_personal_details_online_1": "Credit Card",
                            "enter_personal_details_online_2": "Phone Number",
                            "enter_personal_details_online_3": "Passport",
                            "enter_personal_details_online_4": "Date of Birth",
                            "enter_personal_details_online_5": "Address",
                            "enter_personal_details_online_6": "SSN",
                            "keep_your_passwords_1": "Notepad",
                            "keep_your_passwords_2": "Computer",
                            "keep_your_passwords_3": "Password Manager",
                            "keep_your_passwords_4": "Remember Mentally",
                            "victim_of_online_scam_1": "No",
                            "victim_of_online_scam_2": "Yes",
                            "nline_accounts_hacked_1": "No",
                            "nline_accounts_hacked_2": "Yes",
                        }

                        question_mapping = {
                            "use_the_internet_for_": "What do you use the internet for?",
                            "do_on_social_media_": "What do you do on social media?",
                            "enter_personal_details_online_": "Do you enter personal details online?",
                            "keep_your_passwords_": "How do you keep your passwords?",
                            "victim_of_online_scam_": "Victim of online scam?",
                            "nline_accounts_hacked_": "Account hacked before?"
                        }

                        importances_detail["Display"] = importances_detail["Feature"].map(answer_mapping)
                        importances_detail["Question"] = importances_detail["Feature"].apply(
                            lambda x: next((question_mapping[p] for p in prefixes if x.startswith(p)), "")
                        )

                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        colors = plt.cm.Blues(np.linspace(0.6, 0.95, len(importances_detail)))
                        display_labels = importances_detail["Display"].fillna(importances_detail["Feature"])
                        ax.barh(display_labels, importances_detail["Importance"], color=colors)

                        ax.set_xlabel("Importance Score", fontsize=10, color="#555")
                        fig, ax = set_plot_style(fig, ax)

                        for i, v in enumerate(importances_detail["Importance"]):
                            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color="#555")

                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)

                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_main[1]:
                        top_answer = importances_detail.iloc[0]["Display"]
                        related_question = importances_detail.iloc[0]["Question"]
                        st.markdown(f"""
                            <div class="insight-box">
                                <div class="insight-title">Top Answer Insight</div>
                                <div class="insight-details">
                                    <strong>Most Influential Answer:</strong><br>
                                    {top_answer}<br><br>
                                    <strong>Related Question:</strong><br>
                                    "{related_question}"
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Not enough data for Specific Answers analysis. Minimum 30 records required.")



            # ====== Conversion Rate by Campaign ======
            st.markdown("<div class='section-title'>Conversion Rate by Campaign</div>", unsafe_allow_html=True)

            # Create columns for chart and insight
            conv_rate_cols = st.columns([3, 2])

            with conv_rate_cols[0]:
                # Data Preparation
                all_campaigns = (
                    df.groupby('Campaign number')
                    .agg(num_exposed=('ruserid', 'count'), num_purchases=('purcheas_ind', 'sum'))
                    .reset_index()
                )

                all_campaigns['conversion_rate'] = (all_campaigns['num_purchases'] / all_campaigns['num_exposed']) * 100
                all_campaigns = all_campaigns[all_campaigns['num_exposed'] > 0].sort_values(by='conversion_rate', ascending=True)

                filtered_campaign = int(selected_camp_number)

                # Bar chart
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(7, 4.5))
                colors = plt.cm.Blues(np.linspace(0.6, 0.95, len(all_campaigns)))

                bars = ax.bar(
                    all_campaigns['Campaign number'].astype(str),
                    all_campaigns['conversion_rate'],
                    color=colors,
                    edgecolor="black",
                    width=0.5
                )

                # Highlight selected campaign in bold blue
                for i, bar in enumerate(bars):
                    campaign_number = all_campaigns.iloc[i]['Campaign number']
                    if campaign_number == filtered_campaign:
                        bar.set_color('#4361EE')  # צבע כחול בוהק לקמפיין שנבחר


                ax.set_xlabel("")
                ax.set_ylabel("Conversion Rate (%)", fontsize=10, color="#555")

                ax.grid(axis='y', linestyle='--', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#DDE1E4')
                ax.spines['bottom'].set_color('#DDE1E4')

                ax.tick_params(axis='y', labelsize=8)
                ax.tick_params(axis='x', labelsize=8, rotation=45)

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        height,
                        f"{height:.1f}%",
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        color="#555"
                    )

                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                ax.set_axisbelow(True)

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("</div>", unsafe_allow_html=True)

            with conv_rate_cols[1]:
                # Insight box like in Journey section
                selected_rate = all_campaigns[all_campaigns['Campaign number'] == filtered_campaign]['conversion_rate'].values[0]
                campaign_rank = all_campaigns.reset_index().index[all_campaigns['Campaign number'] == filtered_campaign].tolist()[0] + 1
                total_campaigns = all_campaigns.shape[0]

                st.markdown(f"""
                    <div class="insight-box">
                        <div class="insight-title">Conversion Rate Insight</div>
                        <div class="insight-metric">{selected_rate:.1f}%</div>
                        <div class="insight-details">
                            Campaign <strong>{filtered_campaign}</strong> ranks <strong>{campaign_rank} out of {total_campaigns}</strong> campaigns in conversion performance.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
