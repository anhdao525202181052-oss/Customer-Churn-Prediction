from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title='Olist Customer Churn Demo', page_icon='🛒', layout='wide')
MODEL_DIR=Path('models')
FEATURES=['Frequency','Monetary','Avg_Freight','Avg_Items_Per_Order','Avg_Products_Per_Order','Avg_Category_Diversity','Avg_Review_Score','Avg_Installments']

st.markdown('''<style>
.block-container{padding-top:1.4rem}.risk-high{padding:16px;border-radius:10px;background:#fff1f1;border:1px solid #e89494;color:#971b1b;font-weight:700}.risk-low{padding:16px;border-radius:10px;background:#eefaf2;border:1px solid #87c99d;color:#166b37;font-weight:700}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner='Đang khởi tạo mô hình...')
def build_engine():
    p=MODEL_DIR/'customer_scored.pkl'
    if not p.exists():
        raise FileNotFoundError('Thiếu models/customer_scored.pkl trong GitHub repository.')
    c=pd.read_pickle(p)
    X=c[FEATURES].copy(); y=c['Churn_Risk'].astype(int)
    med=X.median(numeric_only=True); Xf=X.fillna(med)
    model=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,eval_metric='logloss',random_state=42,n_jobs=-1)
    model.fit(Xf,y)
    rfm=np.log1p(c[['Recency','Frequency','Monetary']])
    scaler=StandardScaler().fit(rfm)
    km=KMeans(n_clusters=4,random_state=42,n_init=10).fit(scaler.transform(rfm))
    prof=c.assign(Cluster=km.labels_).groupby('Cluster').agg(Recency=('Recency','mean'),Frequency=('Frequency','mean'),Monetary=('Monetary','mean')).reset_index()
    recent=int(prof.sort_values('Recency').iloc[0].Cluster)
    loyal=int(prof.sort_values(['Frequency','Monetary'],ascending=False).iloc[0].Cluster)
    rest=[int(x) for x in prof.Cluster if int(x) not in {recent,loyal}]
    high=int(prof[prof.Cluster.isin(rest)].sort_values('Monetary',ascending=False).iloc[0].Cluster); low=next(x for x in rest if x!=high)
    names={recent:'Recent Customers',loyal:'Loyal / Repeat Customers',high:'High-Value Inactive',low:'Low-Value Inactive'}
    c=c.copy(); c['Cluster']=km.labels_; c['Segment']=c.Cluster.map(names)
    return c,model,med,scaler,km,names

c,model,med,scaler,km,names=build_engine()

def predict(vals,recency):
    row=pd.DataFrame([vals],columns=FEATURES).fillna(med)
    prob=float(model.predict_proba(row)[0,1])
    r=np.log1p(pd.DataFrame([[recency,vals[0],vals[1]]],columns=['Recency','Frequency','Monetary']))
    cl=int(km.predict(scaler.transform(r))[0])
    return prob,names[cl]

st.title('🛒 Olist Customer Churn Prediction')
st.caption('RFM + K-Means segmentation • XGBoost churn-risk scoring • Retention recommendations')
predict_tab,dashboard_tab,notes_tab=st.tabs(['1. Dự đoán khách hàng','2. Portfolio dashboard','3. Phương pháp'])

with predict_tab:
    st.subheader('Dự đoán nguy cơ rời bỏ của một khách hàng')
    d=c[FEATURES+['Recency']].median(numeric_only=True)
    a,b,d3=st.columns(3)
    recency=a.number_input('Recency (ngày, chỉ dùng cho phân khúc)',min_value=0,value=int(d.Recency),help='Recency không được đưa vào XGBoost vì được dùng để tạo Churn_Risk.')
    freq=b.number_input('Frequency',min_value=1,value=max(1,int(round(d.Frequency))))
    monetary=d3.number_input('Monetary',min_value=0.0,value=float(d.Monetary),step=10.0)
    a,b,d3=st.columns(3)
    freight=a.number_input('Average freight',min_value=0.0,value=float(d.Avg_Freight),step=1.0)
    review=b.slider('Average review score',1.0,5.0,float(np.clip(d.Avg_Review_Score,1,5)),.1)
    installments=d3.number_input('Average installments',min_value=0.0,value=float(d.Avg_Installments),step=.5)
    a,b,d3=st.columns(3)
    items=a.number_input('Average item rows/order',min_value=1.0,value=float(d.Avg_Items_Per_Order),step=.1)
    products=b.number_input('Average products/order',min_value=1.0,value=float(d.Avg_Products_Per_Order),step=.1)
    diversity=d3.number_input('Average category diversity',min_value=0.0,value=float(d.Avg_Category_Diversity),step=.1)
    threshold=st.slider('Classification threshold',.1,.9,.5,.05)
    if st.button('Predict churn risk',type='primary',use_container_width=True):
        vals=[freq,monetary,freight,items,products,diversity,review,installments]
        prob,segment=predict(vals,recency); high=prob>=threshold
        x,y,z=st.columns(3); x.metric('Churn-risk probability',f'{prob:.1%}'); y.metric('Risk label','HIGH RISK' if high else 'LOW RISK'); z.metric('Customer segment',segment)
        st.progress(prob)
        st.markdown('<div class="risk-high">Ưu tiên đưa khách hàng vào chiến dịch giữ chân có kiểm soát.</div>' if high else '<div class="risk-low">Duy trì chăm sóc tiêu chuẩn; tránh ưu đãi đại trà chi phí cao.</div>',unsafe_allow_html=True)
        actions={'High-Value Inactive':'Win-back cá nhân hóa; kiểm tra trải nghiệm; thử nghiệm ưu đãi shipping/voucher có giới hạn.','Low-Value Inactive':'Remarketing tự động chi phí thấp; voucher có ngưỡng chi tiêu.','Recent Customers':'Thúc đẩy lần mua thứ hai bằng post-purchase journey và cross-sell.','Loyal / Repeat Customers':'Loyalty, cross-sell, referral và bảo vệ trải nghiệm.'}
        st.write('**Gợi ý hành động:**',actions[segment])
        st.info('Avg_Freight có khả năng là temporal proxy. Prediction là tín hiệu sàng lọc, không phải bằng chứng quan hệ nhân quả.')

with dashboard_tab:
    st.subheader('Portfolio dashboard')
    scored=c.copy(); scored['Risk_Probability']=model.predict_proba(scored[FEATURES].fillna(med))[:,1]; scored['Predicted_Risk']=np.where(scored.Risk_Probability>=.5,'High','Low')
    m1,m2,m3,m4=st.columns(4); m1.metric('Customers',f'{len(scored):,}'); m2.metric('Predicted high risk',f'{(scored.Predicted_Risk=="High").mean():.1%}'); m3.metric('Average probability',f'{scored.Risk_Probability.mean():.1%}'); m4.metric('Segments',scored.Segment.nunique())
    left,right=st.columns(2)
    seg=scored.groupby('Segment',as_index=False).agg(Customers=('customer_unique_id','count'),Avg_Risk=('Risk_Probability','mean'))
    left.plotly_chart(px.bar(seg,x='Segment',y='Customers',color='Avg_Risk',color_continuous_scale='Reds',title='Customers and average risk by segment'),use_container_width=True)
    right.plotly_chart(px.histogram(scored,x='Risk_Probability',nbins=30,color='Predicted_Risk',title='Risk probability distribution'),use_container_width=True)
    view=scored[['customer_unique_id','Segment','Risk_Probability','Predicted_Risk','Frequency','Monetary','Avg_Freight']].sort_values('Risk_Probability',ascending=False)
    st.dataframe(view.head(500),use_container_width=True,hide_index=True)
    st.download_button('Download scored customer list',view.to_csv(index=False).encode('utf-8'),'olist_customer_risk_scores.csv','text/csv',use_container_width=True)

with notes_tab:
    st.subheader('Phương pháp và phạm vi sử dụng')
    st.markdown('''
- Target vận hành: `Churn_Risk = 1` nếu `Recency > 180 ngày`.
- `Recency`, `Cluster`, `Segment` không được đưa vào XGBoost.
- XGBoost: 300 trees, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8.
- RFM được log1p + StandardScaler trước K-Means (K=4).
- Prototype phục vụ minh họa học thuật. Cần out-of-time validation và A/B testing trước triển khai thực tế.
''')
