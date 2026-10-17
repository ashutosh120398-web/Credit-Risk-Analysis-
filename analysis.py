import pandas as pd, numpy as np, json, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep'); plt.rcParams['figure.dpi']=130
R={}
raw=pd.read_csv('data/credit_risk_dataset.csv')
R['raw_rows']=len(raw); R['missing']=raw.isna().sum()[lambda s:s>0].to_dict(); R['dups']=int(raw.duplicated().sum())

# ---- cleaning
df=raw.drop_duplicates().copy()
n0=len(df)
df=df[(df.person_age<=100)]
df.loc[df.person_emp_length>60,'person_emp_length']=np.nan
R['dropped_age']=n0-len(df)
df['emp_length_missing']=df.person_emp_length.isna().astype(int)
df['int_rate_missing']=df.loan_int_rate.isna().astype(int)
df['person_emp_length']=df.person_emp_length.fillna(df.person_emp_length.median())
df['loan_int_rate']=df.groupby('loan_grade').loan_int_rate.transform(lambda s:s.fillna(s.median()))
df=df.reset_index(drop=True)
df.to_csv('data/credit_risk_cleaned.csv',index=False)
R['clean_rows']=len(df); R['default_rate']=float(df.loan_status.mean())
R['n_default']=int(df.loan_status.sum())

# ---- features for EDA
df['age_band']=pd.cut(df.person_age,[19,24,29,34,44,100],labels=['20-24','25-29','30-34','35-44','45+'])
df['income_band']=pd.qcut(df.person_income,5,labels=['Q1 (lowest)','Q2','Q3','Q4','Q5 (highest)'])
df['lti_band']=pd.cut(df.loan_percent_income,[-.01,.1,.2,.3,.4,1],labels=['0-10%','10-20%','20-30%','30-40%','40%+'])
df['rate_band']=pd.cut(df.loan_int_rate,[0,8,11,14,17,30],labels=['<8%','8-11%','11-14%','14-17%','17%+'])
df['emp_band']=pd.cut(df.person_emp_length,[-1,1,3,6,10,100],labels=['0-1','2-3','4-6','7-10','10+'])

def rate(col): 
    t=df.groupby(col,observed=True).loan_status.agg(['mean','count']); t['mean']*=100; return t
tabs={}
for c in ['loan_grade','person_home_ownership','loan_intent','cb_person_default_on_file','age_band','income_band','lti_band','rate_band','emp_band']:
    tabs[c]=rate(c); R['tab_'+c]={str(k):[round(v['mean'],1),int(v['count'])] for k,v in tabs[c].iterrows()}

# ---- charts
def bar(col,title,fname,order=None,xlabel=None,rot=0):
    t=tabs[col]; 
    fig,ax=plt.subplots(figsize=(7,4))
    sns.barplot(x=t.index.astype(str),y=t['mean'],ax=ax,color='#2E5C8A')
    for i,v in enumerate(t['mean']): ax.text(i,v+0.8,f'{v:.1f}%',ha='center',fontsize=9)
    ax.axhline(df.loan_status.mean()*100,ls='--',c='crimson',lw=1); ax.text(len(t)-0.5,df.loan_status.mean()*100+1,'Overall 21.9%' if False else f'Overall {df.loan_status.mean()*100:.1f}%',color='crimson',ha='right',fontsize=8)
    ax.set_ylabel('Default rate (%)'); ax.set_xlabel(xlabel or col); ax.set_title(title,fontsize=11,weight='bold'); plt.xticks(rotation=rot)
    ax.set_ylim(0,max(t['mean'])*1.18); plt.tight_layout(); plt.savefig('charts/'+fname); plt.close()
bar('loan_grade','Default rate by loan grade','03_default_by_grade.png',xlabel='Loan grade (A = safest)')
bar('person_home_ownership','Default rate by home ownership','04_default_by_home.png',xlabel='Home ownership')
bar('loan_intent','Default rate by loan purpose','05_default_by_intent.png',xlabel='Loan purpose',rot=20)
bar('cb_person_default_on_file','Default rate by prior default on file','06_default_by_prior.png',xlabel='Prior default on file (N/Y)')
bar('lti_band','Default rate by loan-to-income ratio','07_default_by_lti.png',xlabel='Loan amount as % of income')
bar('income_band','Default rate by income quintile','08_default_by_income.png',xlabel='Income quintile')
bar('rate_band','Default rate by interest rate band','09_default_by_rate.png',xlabel='Interest rate')
bar('age_band','Default rate by age band','10_default_by_age.png',xlabel='Age band')

fig,ax=plt.subplots(figsize=(5,4)); c=df.loan_status.map({0:'Repaid',1:'Defaulted'}).value_counts()
ax.pie(c,labels=c.index,autopct='%1.1f%%',colors=['#2E5C8A','#C0392B'],startangle=90,wedgeprops=dict(width=.45)); ax.set_title('Loan outcome split',weight='bold'); plt.tight_layout(); plt.savefig('charts/01_target_split.png'); plt.close()

num=['person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']
fig,axs=plt.subplots(2,4,figsize=(13,6))
for a,cn in zip(axs.flat,num):
    sns.histplot(df[cn],bins=40,ax=a,color='#2E5C8A'); a.set_title(cn,fontsize=9); a.set_xlabel('')
axs.flat[-1].axis('off'); plt.suptitle('Distribution of numeric variables',weight='bold'); plt.tight_layout(); plt.savefig('charts/02_distributions.png'); plt.close()

fig,axs=plt.subplots(1,3,figsize=(12,4))
for a,cn in zip(axs,['loan_int_rate','loan_percent_income','person_income']):
    d=df[df.person_income<250000] if cn=='person_income' else df
    sns.boxplot(x='loan_status',y=cn,data=d,ax=a,palette=['#2E5C8A','#C0392B']); a.set_xticklabels(['Repaid','Defaulted']); a.set_xlabel('')
    a.set_title(cn+' by outcome',fontsize=10)
plt.tight_layout(); plt.savefig('charts/11_boxplots.png'); plt.close()

enc=pd.get_dummies(df[num+['loan_status']+['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']],drop_first=False)
df['grade_num']=df.loan_grade.map({g:i for i,g in enumerate('ABCDEFG')}); df['prior_num']=(df.cb_person_default_on_file=='Y').astype(int)
cm=df[num+['grade_num','prior_num','loan_status']].corr()
fig,ax=plt.subplots(figsize=(8,6.5)); sns.heatmap(cm,annot=True,fmt='.2f',cmap='RdBu_r',center=0,ax=ax,annot_kws={'size':8}); ax.set_title('Correlation matrix',weight='bold'); plt.tight_layout(); plt.savefig('charts/12_correlation.png'); plt.close()
R['corr_target']=cm['loan_status'].drop('loan_status').round(3).sort_values(ascending=False).to_dict()

# heatmap grade x lti
hp=df.pivot_table(index='loan_grade',columns='lti_band',values='loan_status',aggfunc='mean',observed=True)*100
fig,ax=plt.subplots(figsize=(7,4.5)); sns.heatmap(hp,annot=True,fmt='.0f',cmap='YlOrRd',ax=ax,cbar_kws={'label':'Default rate %'}); ax.set_title('Default rate: loan grade x loan-to-income',weight='bold'); ax.set_xlabel('Loan as % of income'); ax.set_ylabel('Loan grade'); plt.tight_layout(); plt.savefig('charts/13_heat_grade_lti.png'); plt.close()
hp2=df.pivot_table(index='person_home_ownership',columns='loan_intent',values='loan_status',aggfunc='mean',observed=True)*100
fig,ax=plt.subplots(figsize=(9,3.8)); sns.heatmap(hp2,annot=True,fmt='.0f',cmap='YlOrRd',ax=ax,cbar_kws={'label':'Default rate %'}); ax.set_title('Default rate: home ownership x loan purpose',weight='bold'); ax.set_xlabel(''); ax.set_ylabel(''); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig('charts/14_heat_home_intent.png'); plt.close()

# ---- risk flag score
df['f_grade']=(df.grade_num>=3).astype(int)           # D-G
df['f_lti']=(df.loan_percent_income>=0.30).astype(int)
df['f_rent']=df.person_home_ownership.isin(['RENT','OTHER']).astype(int)
df['f_prior']=df.prior_num
df['f_rate']=(df.loan_int_rate>=14).astype(int)
df['f_lowinc']=(df.person_income<df.person_income.quantile(.25)).astype(int)
flags=['f_grade','f_lti','f_rent','f_prior','f_rate','f_lowinc']
df['flags']=df[flags].sum(axis=1)
ft=df.groupby('flags').loan_status.agg(['mean','count']); ft['mean']*=100
R['flags']={int(k):[round(v['mean'],1),int(v['count'])] for k,v in ft.iterrows()}
fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(x=ft.index,y=ft['mean'],ax=ax,color='#C0392B')
for i,(v,n) in enumerate(zip(ft['mean'],ft['count'])): ax.text(i,v+1,f'{v:.0f}%\n(n={n})',ha='center',fontsize=8)
ax.set_ylim(0,max(ft['mean'])*1.2); ax.set_xlabel('Number of risk flags'); ax.set_ylabel('Default rate (%)'); ax.set_title('Cumulative risk flags vs default rate',weight='bold'); plt.tight_layout(); plt.savefig('charts/15_risk_flags.png'); plt.close()

# ---- modelling
cat=['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']
X=pd.get_dummies(df[num+cat+['emp_length_missing','int_rate_missing']],columns=cat,drop_first=True); y=df.loan_status
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
sc=StandardScaler().fit(Xtr[num]); Xtr_s=Xtr.copy(); Xte_s=Xte.copy(); Xtr_s[num]=sc.transform(Xtr[num]); Xte_s[num]=sc.transform(Xte[num])
models={'Logistic Regression':(LogisticRegression(max_iter=2000,class_weight='balanced'),True),
        'Random Forest':(RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1),False),
        'Gradient Boosting':(GradientBoostingClassifier(random_state=42),False)}
res={}; plt.figure(figsize=(6,5))
for n,(m,s) in models.items():
    m.fit(Xtr_s if s else Xtr,ytr); xt=Xte_s if s else Xte
    p=m.predict_proba(xt)[:,1]; pr=(p>=.5).astype(int)
    res[n]=dict(AUC=roc_auc_score(yte,p),Accuracy=accuracy_score(yte,pr),Precision=precision_score(yte,pr),Recall=recall_score(yte,pr),F1=f1_score(yte,pr))
    f,t,_=roc_curve(yte,p); plt.plot(f,t,label=f'{n} (AUC {res[n]["AUC"]:.3f})')
plt.plot([0,1],[0,1],'k--',lw=.8); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('ROC curves',weight='bold'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig('charts/16_roc.png'); plt.close()
R['models']={k:{a:round(b,3) for a,b in v.items()} for k,v in res.items()}
best=max(res,key=lambda k:res[k]['AUC']); R['best']=best
gb=models['Gradient Boosting'][0]
cmx=confusion_matrix(yte,(gb.predict_proba(Xte)[:,1]>=.5).astype(int)); R['cm']=cmx.tolist()
fig,ax=plt.subplots(figsize=(4.5,3.8)); sns.heatmap(cmx,annot=True,fmt='d',cmap='Blues',cbar=False,xticklabels=['Repaid','Default'],yticklabels=['Repaid','Default'],ax=ax); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Gradient Boosting confusion matrix',fontsize=10,weight='bold'); plt.tight_layout(); plt.savefig('charts/17_confusion.png'); plt.close()
# importance: group dummies back via permutation on original features
rf=models['Random Forest'][0]
pi=permutation_importance(gb,Xte,yte,scoring='roc_auc',n_repeats=5,random_state=42,n_jobs=-1)
imp=pd.Series(pi.importances_mean,index=X.columns)
grp={}
for c,v in imp.items():
    k=next((g for g in cat if c.startswith(g)),c); grp[k]=grp.get(k,0)+v
imp=pd.Series(grp).sort_values(ascending=True).tail(10)
fig,ax=plt.subplots(figsize=(7,4.5)); imp.plot.barh(ax=ax,color='#2E5C8A'); ax.set_xlabel('Drop in AUC when shuffled'); ax.set_title('Feature importance (permutation, Gradient Boosting)',weight='bold',fontsize=10); plt.tight_layout(); plt.savefig('charts/18_importance.png'); plt.close()
R['importance']={k:round(v,4) for k,v in imp.sort_values(ascending=False).items()}
lr=models['Logistic Regression'][0]; co=pd.Series(lr.coef_[0],index=X.columns); R['lr_top']={k:round(np.exp(v),2) for k,v in co.reindex(co.abs().sort_values(ascending=False).index).head(8).items()}
# business metrics
R['income_med']={'default':float(df[df.loan_status==1].person_income.median()),'repaid':float(df[df.loan_status==0].person_income.median())}
R['lti_med']={'default':float(df[df.loan_status==1].loan_percent_income.median()),'repaid':float(df[df.loan_status==0].loan_percent_income.median())}
R['rate_med']={'default':float(df[df.loan_status==1].loan_int_rate.median()),'repaid':float(df[df.loan_status==0].loan_int_rate.median())}
R['loan_total']=float(df.loan_amnt.sum()); R['loan_default_amt']=float(df[df.loan_status==1].loan_amnt.sum())
json.dump(R,open('results.json','w'),indent=1,default=str)
df.to_csv('data/credit_risk_featured.csv',index=False)
print(json.dumps(R,indent=1,default=str))
