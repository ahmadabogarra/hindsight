#!/usr/bin/env python3
"""Isolated no-secrets Arabic/English embedding comparison on synthetic Odoo policies."""
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

docs = [
("AR1","يتم دفع فواتير الموردين من الخزنة النقدية الرئيسية بعد اعتماد الفاتورة.","ar"),
("AR2","تُحفظ سعة التخزين للهاتف كخاصية مستقلة لكل متغير من المنتج.","ar"),
("AR3","إرجاع شاشة هاتف تالفة إلى المورد يجب أن يرتبط بأمر الشراء الأصلي.","ar"),
("AR4","تقرير الميزانية العمومية يحتسب القيود المحاسبية المرحلة فقط.","ar"),
("AR5","ينبّه النظام المدير عندما ينخفض مخزون الشواحن إلى أقل من خمس قطع.","ar"),
("AR6","تغيير سعر بيع الهاتف يستلزم موافقة المدير قبل اعتماده.","ar"),
("AR7","النسخ الاحتياطي لقواعد بيانات العملاء يحدث في الساعة الثانية صباحا.","ar"),
("AR8","الفاتورة المرحلة للمورد تُصحَّح بإشعار دائن ولا تُمسح مباشرة.","ar"),
("EN1","Customer cash refunds are paid from the main cash register after authorization.","en"),
("EN2","A phone with a repaired display must be classified as used rather than sealed.","en"),
("EN3","Each individually tracked smartphone must have its serial number recorded.","en"),
("EN4","The accessory warranty period is ninety days from purchase.","en"),
("EN5","Manual inventory write-offs require a signed manager approval.","en"),
("EN6","Reconciliation of the cash drawer is carried out every evening.","en"),
("EN7","A supplier credit note must reference the original vendor bill.","en"),
("EN8","Power bank replenishment is triggered when only three units remain.","en"),
]
qs = [
("AR1","من أي حساب ندفع فاتورة المورد؟","Which account pays the vendor invoice?"),
("AR2","وين بنحفظ سعة تخزين الموبايل؟","Where is a mobile phone's storage capacity recorded?"),
("AR3","بأي مستند منربط ارجاع شاشة للمورد؟","What document should a defective screen returned to its supplier link to?"),
("AR4","هل الميزانية بتشمل القيود غير المرحلة؟","Do draft journal entries count in the balance sheet report?"),
("AR5","متى لازم ننبّه على نقص الشواحن؟","At what stock threshold do we alert about chargers?"),
("AR6","مين لازم يوافق قبل تغيير سعر بيع تلفون؟","Whose approval is required before modifying a phone selling price?"),
("AR7","بأي ساعة بيصير باك اب قواعد بيانات العملاء؟","What time are customer databases backed up?"),
("AR8","كيف منصحح فاتورة مورد بعد الترحيل؟","How should a posted supplier bill be corrected?"),
("EN1","من أي خزنة منرجّع مصاري الزبون؟","Where is a customer's cash refund paid from?"),
("EN2","تلفون تبدّلت شاشته بنعتبره مختوم أو مستعمل؟","Is a phone with a replaced screen classified as sealed or used?"),
("EN3","شو لازم نسجل لتعقب تلفون فردياً؟","What identifier is recorded for a serial-tracked smartphone?"),
("EN4","قديش مدة كفالة الاكسسوارات؟","How long is the accessories warranty?"),
("EN5","مين بوافق على شطب مخزون يدوي؟","What authorization is required for a manual inventory write-off?"),
("EN6","كل امتى منعمل مطابقة للخزنة النقدية؟","How frequently is the cash drawer reconciled?"),
("EN7","بأي فاتورة لازم نربط إشعار الدائن من المورد؟","What must the vendor credit note refer to?"),
("EN8","بعد قديش من الباور بانك منعيد طلب مخزون؟","At what remaining quantity do we reorder power banks?"),
]
docids = [d[0] for d in docs]
passages = [d[1] for d in docs]
queries = [(ref, lang, q) for ref, ar, en in qs for lang,q in [("ar",ar),("en",en)]]

def eval_one(name,model_id,query_prompt):
  start=time.perf_counter()
  model=SentenceTransformer(model_id,device="cpu",trust_remote_code=False)
  load_sec=time.perf_counter()-start
  t=time.perf_counter()
  passage_vec=model.encode(passages,normalize_embeddings=True,batch_size=8,show_progress_bar=False)
  passage_sec=time.perf_counter()-t
  raw_queries=[q for _,_,q in queries]
  t=time.perf_counter()
  if query_prompt:
    query_vec=model.encode(raw_queries,prompt=query_prompt,normalize_embeddings=True,batch_size=8,show_progress_bar=False)
  else:
    query_vec=model.encode(raw_queries,normalize_embeddings=True,batch_size=8,show_progress_bar=False)
  query_sec=time.perf_counter()-t
  similarities=np.asarray(query_vec)@np.asarray(passage_vec).T
  rows=[]
  for i,(target,lang,q) in enumerate(queries):
    ranking=np.argsort(-similarities[i]).tolist()
    rank=ranking.index(docids.index(target))+1
    rows.append({"expected":target,"lang":lang,"doc_language":docs[docids.index(target)][2],"rank":rank,"top1":docids[ranking[0]],"query":q})
  print("MODEL "+json.dumps({"name":name,"model":model_id,"dim":int(np.asarray(passage_vec).shape[1]),"load_s":round(load_sec,2),"embed_documents_s":round(passage_sec,2),"embed_queries_s":round(query_sec,2)}),flush=True)
  for lang in ["ar","en"]:
    group=[r for r in rows if r["lang"]==lang]
    print("SCORE "+json.dumps({"model":name,"language":lang,"n":len(group),"top1":sum(r["rank"]==1 for r in group),"top3":sum(r["rank"]<=3 for r in group),"mrr":round(sum(1/r["rank"] for r in group)/len(group),3)}),flush=True)
  for source_lang in ["ar","en"]:
    group=[r for r in rows if r["doc_language"]!=r["lang"] and r["lang"]==source_lang]
    print("CROSSLINGUAL "+json.dumps({"model":name,"query_language":source_lang,"n":len(group),"top1":sum(r["rank"]==1 for r in group)}),flush=True)
  print("ERRORS "+json.dumps({"model":name,"errors":[r for r in rows if r["rank"]>1]},ensure_ascii=False),flush=True)

eval_one("bge-small-en-v1.5","BAAI/bge-small-en-v1.5",None)
eval_one("harrier-oss-v1-270m","microsoft/harrier-oss-v1-270m","Instruct: Retrieve the relevant company policy or operational fact for the question\nQuery: ")
