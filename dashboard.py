import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Nytia Health — Risk Intelligence Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px; padding: 20px;
        text-align: center; border-left: 4px solid;
        margin: 5px;
    }
    .metric-number { font-size: 2rem; font-weight: 800; margin: 0; }
    .metric-label  { font-size: 0.85rem; color: #9ca3af; margin: 0; }
    .metric-sub    { font-size: 0.75rem; color: #6b7280; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = 'https://drive.google.com/uc?export=download&id=1bl-bRZHlgqOmhm6D6MBsOJ11eccVZbhl'
    df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')
    df.columns = df.columns.str.strip()

    def extract_label(text):
        if pd.isna(text): return 'Unknown'
        text = str(text).strip().replace('\u2019',"'").replace('\u2018',"'")
        if text.startswith('Be careful'):      return 'Be Careful'
        elif text.startswith('Your progress'): return 'Declining'
        elif text.startswith('Keep it up'):    return 'Keep It Up'
        elif text.startswith("You've been"):   return 'Making Progress'
        else:                                  return 'Unknown'

    df['status_label'] = df['status_assessment'].apply(extract_label)

    dif_cols = ['dif_nutri','dif_obesic','dif_sleep','dif_depre',
                'dif_wellr','dif_anti_stress','dif_anti_smoke','dif_move']

    df['domains_in_decline'] = df[dif_cols].apply(
        lambda row: sum(1 for v in row if str(v).strip() == '(-1000)-(-250)'), axis=1)

    def assign_tier(s):
        if s >= 6:   return 'CRITICAL'
        elif s >= 4: return 'HIGH'
        elif s >= 2: return 'MEDIUM'
        else:        return 'LOW'

    df['risk_tier'] = df['domains_in_decline'].apply(assign_tier)
    return df, dif_cols

df, dif_cols = load_data()

TIER_COLORS   = {'CRITICAL':'#B71C1C','HIGH':'#E53935','MEDIUM':'#FB8C00','LOW':'#43A047'}
STATUS_COLORS = {'Be Careful':'#E53935','Declining':'#FB8C00',
                 'Keep It Up':'#43A047','Making Progress':'#1E88E5'}
TIER_ORDER    = ['CRITICAL','HIGH','MEDIUM','LOW']
tier_counts   = df['risk_tier'].value_counts()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🏥 Nytia Health")
st.sidebar.markdown("**Risk Intelligence Dashboard**")
st.sidebar.markdown("---")
st.sidebar.markdown("ALY 6980 Capstone — Group 2")
st.sidebar.markdown("Spring 2026 | Northeastern University")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Executive Summary",
    "🎯 Risk Tier Analysis",
    "🏥 Health Domain Insights",
    "🔍 Feature Importance (SHAP)",
    "📋 Intervention Routing",
    "🔎 User Explorer"
])

# ════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.title("🏥 Nytia Health — Risk Intelligence Dashboard")
    st.markdown("##### Proactive Health Risk Stratification | 500K Sample | BigQuery ML XGBoost | Spring 2026")
    st.markdown("---")

    st.markdown("### 📌 Key Metrics")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#B71C1C">
            <p class="metric-number" style="color:#EF9A9A">{tier_counts.get('CRITICAL',0):,}</p>
            <p class="metric-label">🔴 CRITICAL Users</p>
            <p class="metric-sub">Response &lt; 24 hours</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#E53935">
            <p class="metric-number" style="color:#EF9A9A">{tier_counts.get('HIGH',0):,}</p>
            <p class="metric-label">🟠 HIGH Risk</p>
            <p class="metric-sub">Within 3 days</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#FB8C00">
            <p class="metric-number" style="color:#FFCC80">{tier_counts.get('MEDIUM',0):,}</p>
            <p class="metric-label">🟡 MEDIUM Risk</p>
            <p class="metric-sub">Within 2 weeks</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#43A047">
            <p class="metric-number" style="color:#A5D6A7">{tier_counts.get('LOW',0):,}</p>
            <p class="metric-label">🟢 LOW Risk</p>
            <p class="metric-sub">Monthly check-in</p></div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#1E88E5">
            <p class="metric-number" style="color:#90CAF9">96.1%</p>
            <p class="metric-label">🤖 Model Accuracy</p>
            <p class="metric-sub">ROC AUC 0.998</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    ca, cb = st.columns(2)

    with ca:
        st.markdown("### Risk Tier Distribution")
        fig = px.pie(
            values=[tier_counts.get(t,0) for t in TIER_ORDER],
            names=TIER_ORDER,
            color=TIER_ORDER,
            color_discrete_map=TIER_COLORS,
            hole=0.4
        )
        fig.update_traces(textposition='outside', textinfo='percent+label')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown("### Status Label Distribution")
        sl = df['status_label'].value_counts()
        fig2 = px.bar(x=sl.index, y=sl.values, color=sl.index,
                      color_discrete_map=STATUS_COLORS,
                      text=[f'{v/len(df)*100:.1f}%' for v in sl.values])
        fig2.update_traces(textposition='outside')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                           showlegend=False, xaxis_title='Status',
                           yaxis_title='Users', height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Key Business Findings")
    i1,i2,i3 = st.columns(3)
    with i1:
        st.info("**96.05%** of users are 'Be Careful' — severe class imbalance. Engineered risk tiers replace this with actionable 4-level prioritization.")
    with i2:
        st.warning("**Trajectory direction** (dif_*) completely dominates current status (c_val_*). All 8 trajectory features are the top 8 SHAP predictors.")
    with i3:
        st.success("**XGBoost model** achieves 96.1% accuracy and 0.998 ROC AUC — identifying 97-98 out of every 100 at-risk users correctly.")

# ════════════════════════════════════════════════════════════
# PAGE 2 — RISK TIER ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "🎯 Risk Tier Analysis":
    st.title("🎯 Risk Tier Analysis")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Risk Tier Breakdown")
        tier_df = pd.DataFrame({
            'Tier':       TIER_ORDER,
            'Users':      [tier_counts.get(t,0) for t in TIER_ORDER],
            'Percentage': [f"{tier_counts.get(t,0)/len(df)*100:.2f}%" for t in TIER_ORDER],
            'Response':   ['< 24 hours','Within 3 days','Within 2 weeks','Monthly'],
        })
        st.dataframe(tier_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### Domains in Decline")
        dc = df['domains_in_decline'].value_counts().sort_index()
        fig = px.bar(x=dc.index, y=dc.values,
                     color=dc.index, color_continuous_scale='RdYlGn_r',
                     text=[f'{v/len(df)*100:.1f}%' for v in dc.values])
        fig.update_traces(textposition='outside')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                          xaxis_title='Domains in Decline', yaxis_title='Users',
                          showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Risk Tier × Status Label")
    cross = pd.crosstab(df['risk_tier'], df['status_label']).reindex(TIER_ORDER)
    fig3 = px.imshow(cross, color_continuous_scale='YlOrRd', text_auto=True, aspect='auto')
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=320)
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — HEALTH DOMAIN INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "🏥 Health Domain Insights":
    st.title("🏥 Health Domain Insights")
    st.markdown("---")

    domain = st.selectbox("Select Domain:",
                          options=dif_cols,
                          format_func=lambda x: x.replace('dif_','').replace('_',' ').title())

    dif_order = ['(-1000)-(-250)','(-250)-0','0-250','250-1000']
    c1, c2 = st.columns(2)

    with c1:
        vc = df[domain].value_counts().reindex(dif_order, fill_value=0)
        fig = px.bar(x=['Very Low','Low','Good','Excellent'], y=vc.values,
                     color=['Very Low','Low','Good','Excellent'],
                     color_discrete_map={'Very Low':'#B71C1C','Low':'#FB8C00',
                                         'Good':'#43A047','Excellent':'#1E88E5'},
                     text=[f'{v/len(df)*100:.1f}%' for v in vc.values],
                     title=f'{domain} — Trajectory Distribution')
        fig.update_traces(textposition='outside')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                          showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cross2 = pd.crosstab(df[domain], df['risk_tier']).reindex(dif_order).fillna(0)
        fig2 = px.bar(cross2, barmode='group',
                      color_discrete_map=TIER_COLORS,
                      title=f'{domain} — By Risk Tier')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Worst Band Concentration — All Domains")
    worst = {c.replace('dif_','').replace('_',' ').title():
             (df[c]=='(-1000)-(-250)').sum()/len(df)*100 for c in dif_cols}
    ws = pd.Series(worst).sort_values(ascending=True)
    fig3 = px.bar(x=ws.values, y=ws.index, orientation='h',
                  color=ws.values, color_continuous_scale='RdYlGn_r',
                  text=[f'{v:.1f}%' for v in ws.values])
    fig3.update_traces(textposition='outside')
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                       xaxis_title='% Users in Worst Band', showlegend=False, height=380)
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE IMPORTANCE (SHAP)
# ════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance (SHAP)":
    st.title("🔍 Feature Importance — SHAP Analysis")
    st.markdown("##### BigQuery ML Global SHAP from ML.GLOBAL_EXPLAIN")
    st.markdown("---")

    shap_data = {
        'dif_sleep':0.204,'dif_anti_stress':0.187,'dif_depre':0.182,
        'dif_move':0.165,'dif_wellr':0.134,'dif_anti_smoke':0.098,
        'dif_obesic':0.021,'dif_nutri':0.009,
        'c_val_sle':0.0008,'c_val_dep':0.0007,'c_val_anti_stress':0.0006,
        'c_val_move':0.0005,'c_val_wel':0.0004,'c_val_anti_smoke':0.0003,
        'c_val_obe':0.0002,'c_val_nut':0.0001
    }
    shap_df = pd.DataFrame({'Feature':list(shap_data.keys()),
                             'SHAP':list(shap_data.values())}).sort_values('SHAP')
    shap_df['Type'] = shap_df['Feature'].apply(
        lambda x: 'Trajectory dif_*' if x.startswith('dif') else 'Current Status c_val_*')

    fig = px.bar(shap_df, x='SHAP', y='Feature', orientation='h',
                 color='Type',
                 color_discrete_map={'Trajectory dif_*':'#E53935',
                                     'Current Status c_val_*':'#90A4AE'},
                 text=[f'{v:.4f}' for v in shap_df['SHAP']])
    fig.update_traces(textposition='outside')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                      xaxis_title='SHAP Attribution Value', height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info("""
    **Key Finding: Trajectory completely dominates current status.**
    
    All 8 dif_* trajectory columns are the top 8 predictors.
    All 8 c_val_* current status columns are the bottom 8 — all below 0.001 SHAP value.
    
    **This means: where a user is heading matters far more than where they currently are.**
    """)

# ════════════════════════════════════════════════════════════
# PAGE 5 — INTERVENTION ROUTING
# ════════════════════════════════════════════════════════════
elif page == "📋 Intervention Routing":
    st.title("📋 Intervention Routing Framework")
    st.markdown("---")

    int_df = pd.DataFrame({
        'Risk Tier':    ['🔴 CRITICAL','🟠 HIGH','🟡 MEDIUM','🟢 LOW'],
        'Users':        [f"{tier_counts.get(t,0):,}" for t in TIER_ORDER],
        '%':            [f"{tier_counts.get(t,0)/len(df)*100:.2f}%" for t in TIER_ORDER],
        'Response SLA': ['< 24 hours','Within 3 days','Within 2 weeks','Monthly'],
        'Strategy':     [
            'Immediate outreach. Assign dedicated health coach.',
            'Proactive telehealth. Structured 90-day plan.',
            'Preventive content. Habit-building resources.',
            'Maintenance check-in. Positive reinforcement.'
        ]
    })
    st.dataframe(int_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### At Full Production Scale (430M Users)")
    scale = 430981696 / 500000
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("🔴 CRITICAL", f"{int(tier_counts.get('CRITICAL',0)*scale):,.0f}")
    with c2: st.metric("🟠 HIGH",     f"{int(tier_counts.get('HIGH',0)*scale):,.0f}")
    with c3: st.metric("🟡 MEDIUM",   f"{int(tier_counts.get('MEDIUM',0)*scale):,.0f}")
    with c4: st.metric("🟢 LOW",      f"{int(tier_counts.get('LOW',0)*scale):,.0f}")

    st.markdown("---")
    st.markdown("### Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        perf = pd.DataFrame({
            'Metric':         ['Accuracy','Recall','ROC AUC','F1 Score','Training Time'],
            'Value':          ['96.1%','97.6%','0.998','96.7%','22 minutes'],
            'Interpretation': [
                '96 out of 100 users correctly classified',
                '97-98 out of 100 at-risk users identified',
                'Near-perfect discrimination between tiers',
                'Balanced precision and recall',
                'Parallel slot time 8h 20min on BigQuery'
            ]
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=96.1,
            title={'text':"Model Accuracy (%)",'font':{'color':'white'}},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':'#43A047'},
                   'steps':[{'range':[0,70],'color':'#B71C1C'},
                             {'range':[70,85],'color':'#FB8C00'},
                             {'range':[85,100],'color':'#1B5E20'}]},
            number={'font':{'color':'white'}}
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=280)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 6 — USER EXPLORER
# ════════════════════════════════════════════════════════════
elif page == "🔎 User Explorer":
    st.title("🔎 User Explorer")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        sel_tier = st.selectbox("Filter by Risk Tier:", ['All']+TIER_ORDER)
    with c2:
        sel_status = st.selectbox("Filter by Status:",
                                  ['All','Be Careful','Declining','Keep It Up','Making Progress'])

    filtered = df.copy()
    if sel_tier   != 'All': filtered = filtered[filtered['risk_tier']   == sel_tier]
    if sel_status != 'All': filtered = filtered[filtered['status_label']== sel_status]

    st.markdown(f"**Showing {len(filtered):,} users**")
    display_cols = dif_cols + ['status_label','risk_tier','domains_in_decline']
    st.dataframe(filtered[display_cols].head(200), use_container_width=True, hide_index=True)

    if sel_tier != 'All':
        st.markdown(f"### Domain Profile — {sel_tier} Tier")
        worst = {c.replace('dif_','').replace('_',' ').title():
                 (filtered[c]=='(-1000)-(-250)').sum()/len(filtered)*100 for c in dif_cols}
        ws = pd.Series(worst).sort_values(ascending=False)
        fig = px.bar(x=ws.index, y=ws.values, color=ws.values,
                     color_continuous_scale='RdYlGn_r',
                     text=[f'{v:.1f}%' for v in ws.values])
        fig.update_traces(textposition='outside')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                          xaxis_title='Health Domain', yaxis_title='% in Worst Band',
                          showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
