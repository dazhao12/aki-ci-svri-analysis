"""CI-SVRI phenotype analysis. Requires an authorised analysis-ready cohort.

No participant data are distributed. AKI_7d and eGFR are accepted as the
author-confirmed analysis inputs; this program does not adjudicate outcomes.
The estimation functions below are extracted unchanged from the adopted code.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse, json
import numpy as np
import pandas as pd
from patsy import dmatrices, build_design_matrices
from scipy.special import expit
from scipy.stats import chi2
from threadpoolctl import threadpool_limits
from firthmodels import FirthLogisticRegression
from firthmodels.logistic import compute_logistic_quantities
from firthmodels._solvers import newton_raphson


ORDER = ['normal_normal', 'high_flow_low_resistance', 'low_flow_high_resistance', 'other_mixed']

PROFILE = "C(preop_state, Treatment(reference='normal_normal'))"

CORE_COVARIATES = [
    "trial_arm",
    "Age",
    "Male",
    "BMI",
    "Smoking",
    "History_of_myocardial_infarction",
    "Arrhythmia",
    "Diabetes",
    "Hypertension",
    "eGFR",
    "Hb",
    "Left_ventricular_EF",
]

PREOP_TERMS = ["preop_hf_lr", "preop_lf_hr", "preop_other"]

def restricted_test(m, indices, values=None):
    """固定指定系数为0；约束拟合仍用完整设计矩阵的Jeffreys惩罚。"""
    fixed = np.array(indices, dtype=int)
    free = np.array([i for i in range(len(m.coef_)) if i not in fixed])
    X, y, weights, offset = m._fit_data
    def quantities(gamma):
        beta = np.zeros(len(m.coef_)); beta[free] = gamma
        if values is not None:beta[fixed]=values
        q = compute_logistic_quantities(X=X, y=y, beta=beta, sample_weight=weights,
            offset=offset, workspace=m._workspace, penalty_weight=m.penalty_weight)
        return SimpleNamespace(loglik=q.loglik, modified_score=q.modified_score[free].copy(),
            fisher_info=q.fisher_info[np.ix_(free, free)].copy())
    r = newton_raphson(quantities, len(free), max_iter=400, max_step=5,
        max_halfstep=30, gtol=1e-6, xtol=1e-7, beta_init=m.coef_[free], warn_on_failure=True)
    assert r.converged, (indices, r.failure_reason)
    lr = max(0., 2*(m.loglik_-r.loglik))
    return float(chi2.sf(lr, len(fixed))), lr

def fit(formula,d):
    y,x=dmatrices(formula,d,return_type='dataframe');info=x.design_info
    means=np.zeros(x.shape[1]);scales=np.ones(x.shape[1])
    for i,c in enumerate(x.columns):
        if x[c].nunique()>2:means[i]=x[c].mean();scales[i]=x[c].std(ddof=0)
    X=(x.to_numpy(float)-means)/scales
    assert np.linalg.matrix_rank(X)==X.shape[1]
    m=FirthLogisticRegression(fit_intercept=False,backend='numpy',max_iter=400,gtol=1e-7,xtol=1e-8).fit(X,y.to_numpy().ravel())
    assert m.converged_ and np.isfinite(m.coef_).all()
    p=expit(X@m.coef_);cov=np.linalg.inv(X.T@(X*(p*(1-p))[:,None]))
    def predict(q):
        xx=(np.asarray(build_design_matrices([info],q)[0])-means)/scales
        pp=expit(xx@m.coef_);r=pp.mean();g=np.mean(xx*(pp*(1-pp))[:,None],axis=0)
        se=np.sqrt(g@cov@g)
        return float(r),float(max(0,r-1.96*se)),float(min(1,r+1.96*se))
    return m,list(x.columns),predict

def effects(m,names):
    ix=[names.index(PROFILE+'[T.'+g+']') for g in ORDER[1:]]
    ci=m.conf_int(method='pl',features=ix,max_iter=200,tol=1e-7)
    rows={}
    for g,i in zip(ORDER[1:],ix):
        pp,lr=restricted_test(m,[i])
        for v in ci[i]:
            _,z=restricted_test(m,[i],[v]);assert abs(z-chi2.ppf(.95,1))<2e-4
        rows[g]=(float(np.exp(m.coef_[i])),*np.exp(ci[i]).tolist(),pp)
    gp,lr=restricted_test(m,ix)
    return rows,gp,lr

def basis(q,v,knots):
    q=q.copy();x=q[v].to_numpy();a,b=knots[-2:];den=(knots[-1]-knots[0])**2
    q['spline_linear']=x
    for j,k in enumerate(knots[:-2]):
        q[f'spline_nl{j}']=(np.maximum(x-k,0)**3-(b-k)/(b-a)*np.maximum(x-a,0)**3+(a-k)/(b-a)*np.maximum(x-b,0)**3)/den
    return q

def classify(d,ci_low=2.5,svri_low=2000,svri_high=2400):
    q=d.copy();ci=q.Cardiac_index_baseline;sv=q.preop_SVRI
    q['preop_state']=np.select([ci.between(ci_low,4,inclusive='both')&sv.between(svri_low,svri_high,inclusive='both'),
        (ci>4)&(sv<svri_low),(ci<ci_low)&(sv>svri_high)],ORDER[:3],default=ORDER[3])
    for col,g in zip(intra.PREOP_TERMS,ORDER[1:]):q[col]=(q.preop_state==g).astype(float)
    return q

SPECS = [
 ('Age', 'Age, per 5 years', 5),
 ('Male', 'Male sex (vs female)', 1),
 ('BMI', 'BMI, per 1 kg/m²', 1),
 ('Smoking', 'Current or former smoking (vs never)', 1),
 ('Hypertension', 'Hypertension (yes vs no)', 1),
 ('Diabetes', 'Diabetes mellitus (yes vs no)', 1),
 ('History_of_myocardial_infarction', 'Previous myocardial infarction (yes vs no)', 1),
 ('Arrhythmia', 'Arrhythmia (yes vs no)', 1),
 ('eGFR', 'eGFR, per 10 mL/min/1.73 m²', 10),
 ('Hb', 'Hemoglobin, per 10 g/L', 10),
 ('Left_ventricular_EF', 'LVEF, per 5 percentage points', 5),
 ('trial_arm', 'Trial group: intervention vs usual care', 1),
]

LABELS = ['High CI/low SVRI', 'Low CI/high SVRI', 'Other combinations']

def collect(formula, d, terms):
    m,names,_ = fit(formula,d)
    _,raw = dmatrices(formula,d,return_type='dataframe')
    assert len(raw)==1949 and np.isfinite(raw.to_numpy()).all()
    ix=[names.index(t[0]) for t in terms]
    ci=m.conf_int(method='pl',features=ix,max_iter=200,tol=1e-7)
    out=[]
    for term,label,unit in terms:
        i=names.index(term)
        scale=raw[term].std(ddof=0) if raw[term].nunique()>2 else 1.
        factor=unit/scale
        p,lr=restricted_test(m,[i])
        for bound in ci[i]:
            _,test=restricted_test(m,[i],[bound])
            assert abs(test-chi2.ppf(.95,1))<2e-4
        vals=np.exp(np.r_[m.coef_[i],ci[i]]*factor)
        assert np.isfinite(vals).all() and vals[1]<vals[0]<vals[2]
        out.append(dict(term=term,label=label,increment=unit,internal_SD=scale,
                        OR=vals[0],lower_95=vals[1],upper_95=vals[2],P=p,LR=lr))
    return out,m,names
def make_candidates(d):
    # Nine reported post hoc subgroup variables. Age retains continuous age adjustment.
    return [
      ('age','年龄','Age',(d.Age>=65).astype(int),None,['<65','≥65']),
      ('sex','性别','Sex',d.Male,'Male',['Female','Male']),
      ('bmi','BMI','BMI',(d.BMI>=28).astype(int),None,['<28','≥28']),
      ('hypertension','高血压','Hypertension',d.Hypertension,'Hypertension',['No','Yes']),
      ('diabetes','糖尿病','Diabetes',d.Diabetes,'Diabetes',['No','Yes']),
      ('egfr','肾功能','eGFR',(d.eGFR<60).astype(int),None,['≥60','<60']),
      ('trial','随机分组','Trial group',d.trial_arm,'trial_arm',['Usual care','Intervention']),
      ('mi','既往心肌梗死','Previous myocardial infarction',d.History_of_myocardial_infarction,'History_of_myocardial_infarction',['No','Yes']),
      ('lvef','左室射血分数','LVEF',(d.Left_ventricular_EF<50).astype(int),None,['≥50','<50'])]

intra=SimpleNamespace(PREOP_TERMS=PREOP_TERMS)
a=SimpleNamespace(**globals())

def run(cohort,output,section):
    W=Path(output)
    W.mkdir(parents=True,exist_ok=True)
    d=classify(pd.read_csv(cohort))
    required=CORE_COVARIATES+['AKI_7d','Cardiac_index_baseline','preop_SVRI']
    assert d[required].notna().all().all(), 'Analysis inputs must be complete.'
    assert len(d)==1949 and d.AKI_7d.sum()==209, 'Expected adopted cohort: 1949 participants, 209 events.'
    adj=' + '.join(a.CORE_COVARIATES)
    rows=[];diagnostics=[]
    for key,cl,sl,sh in [('primary',2.5,2000,2400),('ci_2_2',2.2,2000,2400),('previous_svri',2.5,1970,2390)]:
        q=a.classify(d,cl,sl,sh);m,names,predict=a.fit('AKI_7d ~ '+a.PROFILE+' + '+adj,q)
        rr,gp,lr=a.effects(m,names);ref=q[q.preop_state==a.ORDER[0]]
        for g in a.ORDER:
            z=q[q.preop_state==g];r,lo,hi=predict(q.assign(preop_state=g));e=(1,1,1,np.nan) if g==a.ORDER[0] else rr[g]
            rows.append(dict(model=key,group=g,n=len(z),events=int(z.AKI_7d.sum()),AKI_pct=100*z.AKI_7d.mean(),OR=e[0],lower_95=e[1],upper_95=e[2],P=e[3],adjusted_AKI_pct=100*r,adjusted_lower=100*lo,adjusted_upper=100*hi,CI_lower=cl,CI_upper=4,SVRI_lower=sl,SVRI_upper=sh,ref_n=len(ref),ref_events=int(ref.AKI_7d.sum()),global_P=gp))
        diagnostics.append(dict(model=key,converged=True,rank=len(names),n=len(q),global_P=gp,global_LR=lr))
        print(key,rr[a.ORDER[1]],flush=True)
    pd.DataFrame(rows).to_csv(W/'preop_effects.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([r for r in rows if r['group']==a.ORDER[1]]).to_csv(W/'sensitivity.csv',index=False,encoding='utf-8-sig')
    if section=='primary':return
    # Nine reported subgroup analyses, with fixed cutoffs.
    # Labels are constructed from subgroup definitions, not historical result files.
    specs={s[0]:s for s in a.make_candidates(d)};subrows=[];inter=[]
    keys=['age','sex','bmi','hypertension','diabetes','egfr','trial','mi','lvef']
    for key in ['overall']+keys:
        spec=specs.get(key);q=d.copy() if spec is None else d.assign(subgroup=spec[3].astype(int))
        covs=a.CORE_COVARIATES if spec is None or key=='age' else [c for c in a.CORE_COVARIATES if c!=spec[4]]
        prediction=None;gp=np.nan;cp={g:np.nan for g in a.ORDER[1:]}
        for level in ([-1] if spec is None else [0,1]):
            formula='AKI_7d ~ '+a.PROFILE+(f' * C(subgroup, Treatment(reference={level}))' if spec else '')+' + '+' + '.join(covs)
            m,names,_=a.fit(formula,q);rr,_,_=a.effects(m,names)
            pp=expit(m._fit_data[0]@m.coef_)
            if level==1:assert np.allclose(pp,prediction,atol=1e-7)
            prediction=pp
            if level==0:
                ix=[i for i,t in enumerate(names) if ':' in t];assert len(ix)==3
                gp,lr=a.restricted_test(m,ix)
                inter.append(dict(subgroup=key,scope='overall',df=3,P=gp,statistic=lr))
                for g in a.ORDER[1:]:
                    ixg=[i for i in ix if '[T.'+g+']' in names[i]];assert len(ixg)==1
                    cp[g],lr=a.restricted_test(m,ixg)
                    inter.append(dict(subgroup=key,scope=g,df=1,P=cp[g],statistic=lr))
            z=q if spec is None else q[q.subgroup==level];ref=z[z.preop_state==a.ORDER[0]]
            for g in a.ORDER[1:]:
                rawrow=dict(subgroup=key,level=level,group=g,level_label='Overall' if spec is None else spec[5][level])
                gg=z[z.preop_state==g];o,lo,hi,p=rr[g]
                rawrow.update(n=len(gg),events=int(gg.AKI_7d.sum()),ref_n=len(ref),ref_events=int(ref.AKI_7d.sum()),OR=o,lower_95=lo,upper_95=hi,P=p,global_interaction_P=gp,contrast_interaction_P=cp[g],Firth_OR=o,Firth_lower_95=lo,Firth_upper_95=hi,Firth_effect_P=p,Firth_global_interaction_P=gp,Firth_contrast_interaction_P=cp[g])
                subrows.append(rawrow)
            diagnostics.append(dict(model=key,level=level,converged=True,rank=len(names),n=len(q)))
        pd.DataFrame(subrows).to_csv(W/'subgroups.csv',index=False,encoding='utf-8-sig')
        print('亚组完成',key,flush=True)
    assert len(subrows)==57 and len(inter)==36
    pd.DataFrame(inter).to_csv(W/'interactions.csv',index=False,encoding='utf-8-sig')
    curves=[];tests=[]
    for v,label in [('Cardiac_index_baseline','CI'),('preop_SVRI','SVRI')]:
        knots=d[v].quantile([.05,.35,.65,.95]).to_numpy();q=a.basis(d,v,knots)
        m,names,predict=a.fit('AKI_7d ~ spline_linear + spline_nl0 + spline_nl1 + '+adj,q)
        ix=[names.index(k) for k in ['spline_linear','spline_nl0','spline_nl1']]
        tp,tlr=a.restricted_test(m,ix);npv,nlr=a.restricted_test(m,ix[1:])
        tests.append(dict(exposure=label,total_P=tp,nonlinear_P=npv,total_LR=tlr,nonlinear_LR=nlr,knots=knots.tolist()))
        for x in np.linspace(d[v].quantile(.01),d[v].quantile(.99),161):
            risk,lo,hi=predict(a.basis(d.assign(**{v:x}),v,knots))
            curves.append(dict(exposure='SVRI' if label=='SVRI' else v,x=x,risk=risk,lower=lo,upper=hi))
        print('RCS完成',label,tp,npv,flush=True)
    pd.DataFrame(curves).to_csv(W/'curves.csv',index=False,encoding='utf-8-sig')
    m,names,_=a.fit('AKI_7d ~ Cardiac_index_baseline + '+a.PROFILE+' + '+adj,d)
    ip,ilr=a.restricted_test(m,[names.index(a.PROFILE+'[T.'+g+']') for g in a.ORDER[1:]])
    terms=[(a.PROFILE+'[T.'+g+']',label,1) for g,label in zip(a.ORDER[1:],a.LABELS)]
    multi,mm,mn=a.collect('AKI_7d ~ '+a.PROFILE+' + '+adj,d,terms+a.SPECS)
    uni,um,un=a.collect('AKI_7d ~ '+a.PROFILE,d,terms)
    overall={key:a.restricted_test(m,[ns.index(t[0]) for t in terms])[0] for key,m,ns in [('univariable',um,un),('multivariable',mm,mn)]}
    for spec in a.SPECS:
        rr,_,_=a.collect('AKI_7d ~ '+spec[0],d,[spec]);uni.extend(rr)
    paired=[]
    for u,m in zip(uni,multi):
        assert u['term']==m['term'];r={k:m[k] for k in ['term','label','increment']}
        for prefix,obj in [('univariable',u),('multivariable',m)]:r.update({prefix+'_'+k:obj[k] for k in ['OR','lower_95','upper_95','P']})
        paired.append(r)
    pd.DataFrame(paired).to_csv(W/'regression_table.csv',index=False,encoding='utf-8-sig')

    (W/'model_tests.json').write_text(json.dumps(dict(RCS=tests,incremental_fit=dict(P=ip,LR=ilr,df=3),overall_phenotype_P=overall),indent=2),encoding='utf8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',required=True,type=Path,help='Authorised analysis-ready CSV; not supplied in this repository.')
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--section',choices=['primary','all'],default='primary',help='primary: main model plus two threshold sensitivities; all: also reported subgroups, splines and coefficient table.')
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.cohort,args.output,args.section)
