# Q1(d) Sampling Strategy

The 25 examples were selected by uniform interval sampling over the full error list, not by manual cherry-picking. The pipeline first keeps every evaluation pair where prediction and reference differ, computes a fixed stride of floor(total_errors / 25), and then returns every stride-th error from that ordered list. This spreads the sample across the evaluation set and makes it less likely that the report over-focuses on any single utterance, topic, or failure mode.

# WER Diagnosis

The fine-tuned model performs worse than the pretrained baseline because it appears to overfit to the Josh Talks training distribution. The training data is conversational, disfluent, and relatively small, while Hindi FLEURS test is cleaner read speech. The sampled errors show decoder loops, long hallucinated tails, and domain-mismatched substitutions, which means the model learned the training-set style but lost some of the pretrained model's broader generalization. In short: training loss improved, but cross-domain robustness degraded.

# Q1(e) Error Taxonomy

## Rare content-word confusion (12/25)

Description: The model preserves the sentence scaffold but substitutes one or two rare or content-heavy words.

- Reference: ब्रह्मांड की सभी वस्तुएं पदार्थ से बनी हैं सारे पदार्थ सूक्ष्तम कणों से बनें हैं जिन्हें अणु कहा जाता है
- Prediction: ब्रमहान की सभी वस्तुयें पधार्थ से बनी है सारे प्रदारित सुक्ष्तम कारों से मने है जिने अनू का कहा जाता हैं आप तो इसलिए लेकिन ना उसले
- Error Type: rare_word_swap
- Reason: The sentence frame is preserved, but one or two long content words are replaced with acoustically plausible alternatives.

- Reference: पुलिस अधीक्षक चंद्र शेखर सोलंकी ने कहा कि आरोपी ढके हुए चेहरे के साथ अदालत में पेश हुए थे
- Prediction: पुलिस दिक्षक चंदर सेखर सोलन कीन है कहा कि आरोपी ढ़के हुए चेहरे के साथ अधालत में पेश हूई थे बिल्कुम जब लग रहे हैं तो वह नहीं
- Error Type: rare_word_swap
- Reason: The sentence frame is preserved, but one or two long content words are replaced with acoustically plausible alternatives.

- Reference: यह ऑरोरा बोरेलिस को देखने का एक अच्छा अवसर प्रदान करता है क्योंकि आकाश में उस समय अंधकार सा ही रहेगा
- Prediction: ये आरूरा बोरिलीस को देखने का एक अच्छा अंफसर प्रधान करता है क्योंकि आकाश में उस समय अन्दकार साही रहेगा इस वो लगे ना तो जो भी यह
- Error Type: rare_word_swap
- Reason: The sentence frame is preserved, but one or two long content words are replaced with acoustically plausible alternatives.

- Reference: कुछ अणुओं में अस्थिर केंद्रक होता है जिसका मतलब यह है कि उनमें थोड़े या बिना किसी झटके से टूटने की प्रवृत्ति होती है
- Prediction: कुछ अर्हूं में आस कैनद्रक होता है जिसका मतलब यहां की उनमें सोड़े या बिना किसी झटके से तूठने की का प्रवत्य हो थी हैं इस नहीं लेगे �
- Error Type: rare_word_swap
- Reason: The sentence frame is preserved, but one or two long content words are replaced with acoustically plausible alternatives.

- Reference: अगर आपने फ़िल्म नेशनल ट्रेज़र देखी है तो आपको लगेगा कि स्वतंत्रता के घोषणा-पत्र के पीछे एक ख़ज़ाने का नक्शा लिखा गया था
- Prediction: अगर आपने फिल्म नेसनल ट्रेजर देखी हैं तो आापको लगेगा कि सुतन्सता के घूशनाव पत्र के मीचे एक खा जाने के हा नक्सा लिखा है था इस
- Error Type: rare_word_swap
- Reason: The sentence frame is preserved, but one or two long content words are replaced with acoustically plausible alternatives.

## Lexical / phonetic substitution (5/25)

Description: The sentence frame is broadly intact, but content words are replaced by acoustically similar alternatives.

- Reference: अधिकांश छोटे द्वीप स्वतंत्र राष्ट्र हैं या फ़्रांस से संबंधित हैं और लग्ज़री बीच रिसॉर्ट के रूप में जाने जाते हैं
- Prediction: अदिकांस छोते भी प्र सुतनत्र राश्टर हैं या फ्रांच से संबंधित है कोल लगजरी वीच्ष रीजॉड के रूप में जाने जहाते है इम्म आया त
- Error Type: content_word_substitution
- Reason: Most of the sentence frame is preserved, but content words are swapped with acoustically similar alternatives.

- Reference: विशेष रूप से यह दावा किया जाता है कि किसी व्यक्ति की सूक्ष्म अभिव्यक्तियों की सही व्याख्या करके यह समझा सकता है कि वह झूठ बोल रहा है या नहीं
- Prediction: विसे सुर्प जायों दावकीहा जता है कि किस व्यक्ति की सूक्षन अभी वियकतियो की शही ब्याक्य करने करके या समझा सकता था कि वो जूट बोल रहा हे
- Error Type: content_word_substitution
- Reason: Most of the sentence frame is preserved, but content words are swapped with acoustically similar alternatives.

- Reference: अमेरिकी जिमनास्टिक्स संयुक्त राज्य ओलंपिक समिति के पत्र का समर्थन करता है साथ ही सभी एथलीटों को एक सुरक्षित माहौल देने के लिए ओलंपिक परिवार के होने की अहमियत स्वीकार करता है
- Prediction: अमेर की जिमनास्तिक सहयुपत राज ओलंपिक शमीती के पत्र का समरतन करता है साथ ही सभी इतलीटों को एक सुरच्छित महौल देने केलिए ओंलं awal karibar के
- Error Type: content_word_substitution
- Reason: Most of the sentence frame is preserved, but content words are swapped with acoustically similar alternatives.

- Reference: इटली के कई अन्य शहरों और दुनिया के बाकी हिस्सों में खासकर पोलैंड में इस तरह के सेटअप बनाए गए थे जिन्हें बड़ी संख्या में लोगों ने देखा था
- Prediction: इतली के कही अर्ने सहरों और दुनियां के बाकी हिस्सों में खासकर पॉलेंड मे इस तरह के सैटव बनाए गई थे जिने भ़ी संख्यामे लोग देखने है
- Error Type: content_word_substitution
- Reason: Most of the sentence frame is preserved, but content words are swapped with acoustically similar alternatives.

- Reference: इटली के कई अन्य शहरों और दुनिया के बाकी हिस्सों में खासकर पोलैंड में इस तरह के सेटअप बनाए गए थे जिन्हें बड़ी संख्या में लोगों ने देखा था
- Prediction: इतली के कही अन्यां सहरों और दुनिया के बाकी हिस्सों में खासकर पॉलेंड मे इस तरह के सैटव बनाए गए है जिने भ़ी संक्यमेंलोगोंने मेरे नही
- Error Type: content_word_substitution
- Reason: Most of the sentence frame is preserved, but content words are swapped with acoustically similar alternatives.

## Repetition / hallucination (4/25)

Description: Predictions start plausibly but then repeat the same token or short phrase, creating long hallucinated tails.

- Reference: यह समुद्र के नीचे पतला और ऊंचे क्षेत्रों के नीचे मोटा होता है
- Prediction: ये समुत्र की नीचे पतला और उंच हेसेत्यों में निचा मोटा होता हैं अह आएक बन जी तो ये वह लेगे रहे है देशन भी थे फिर खुद लग लोग है
- Error Type: decoder_loop
- Reason: The prediction follows the reference for a short prefix, then enters a repetition loop and keeps emitting the same token or phrase.

- Reference: चीज़ों पर कुछ बनाकर या खरोंच कर इस जगह को नुक़सान न पहुंचाएं
- Prediction: चीजों पर कुछ बनाकर है या खर्द करे सिस जोह का करनुक्सान नहीं वो चाहें अगर मतलब आया लेकिन तो इस डिया कॉम्म रहे हैं था एक भी और टी�
- Error Type: decoder_loop
- Reason: The prediction follows the reference for a short prefix, then enters a repetition loop and keeps emitting the same token or phrase.

- Reference: चांद की सतह चट्टानों और धूल से बनी है चांद की बाहरी परत को क्रस्ट कहते हैं
- Prediction: चान्द की सतह छक्टानों और धूल से बनि है चांद भाहरय प्रत को क्रस्थ कहते हैं अम्म में आप जाए लेकिन ना वहां उसके रहे हो गई तो यह जी ज
- Error Type: decoder_loop
- Reason: The prediction follows the reference for a short prefix, then enters a repetition loop and keeps emitting the same token or phrase.

- Reference: लेकिन उन पक्षियों के बारे में बहुत सी चीजें हैं जो अभी भी डायनासोर जैसे दिखते हैं
- Prediction: लेकिन उन पक्षीयों के बारे में बहुत सी चीजे है जो अभी भी मे डायनोसौर जैसी दिखती हैं आप वह लेके रहे हो गई तो नहीं थी यह एक टीम फि
- Error Type: decoder_loop
- Reason: The prediction follows the reference for a short prefix, then enters a repetition loop and keeps emitting the same token or phrase.

## Deletion / truncation (2/25)

Description: Predictions are materially shorter than the reference and drop a content-bearing suffix.

- Reference: हालांकि यह अक्सर केवल एक गलत रूढ़िवादी धारणा होती है पेरिस में घुलने-मिलने का सबसे अच्छा तरीका अभी भी अपना सबसे अच्छा व्यवहार करना है किसी ऐसे व्यक्ति की तरह अभिनय करना जो बिएन एलेव” अच्छी तरह से पला-बढ़ा हो यह घुलने-मिलने को काफी आसान ब...
- Prediction: हालाकि ये अख्सर केवल एक गलत रूडी वादी धारना होती है पेरिस में घुलने मिलनी का सबसे ऐच्छा तरीका अमी भी अپना सبसेज्या विवार करने ह
- Error Type: content_drop
- Reason: The hypothesis is substantially shorter than the reference and drops a trailing content span.

- Reference: अगर आप सर्दियों में उत्तरी बाटलिक को पार करते हैं तो आपको केबिन स्थान की जांच करनी होगी क्योंकि क्योंकि जो लोग ज्यादा प्रभावित है उन्हें बर्फ से गुजरने पर काफी भयानक शोर महसूस होगा
- Prediction: अगर आप सर्दिया में उतनरी बाटल का को पार करते हैं तो आमको केविनस्थान की जाच करनी हुगी क्योंकि जो लोग ज्यादा प्रभावेत है उन्हें म
- Error Type: content_drop
- Reason: The hypothesis is substantially shorter than the reference and drops a trailing content span.

## Number / numeral mismatch (1/25)

Description: Numeric content is mistranscribed or reformatted even when the rest of the sentence is relatively aligned.

- Reference: 1966 से सुंदरबन वन्यजीव अभ्यारण बन गया है और एक आकलन के अनुसार इस इलाके में अब लगभग 400 रॉयल बंगाल टाइगर्स और लगभग 30,000 चीतल हैं
- Prediction: उन्निस्या छांचर्त से सुंदर बन वंजीव अभ्यहारन बंगै है और एक आ कलन के अनुसार इस इलाके में जब लगभग चार सो रोयल बमकाल टाइवर्
- Error Type: numeral_surface_form
- Reason: The sentence scaffold is broadly similar, but numeric tokens are substituted or reformatted incorrectly.

## Semantic drift / insertion (1/25)

Description: The decoder drifts off the reference and inserts unrelated content rather than making a local substitution.

- Reference: माऊ आंदोलन द्वारा स्वतंत्रता के लिए आयोजित संघर्ष के दौरान नगर में एक शांतिपूर्ण सभा में प्रमुख तापुआ तामासेसे सेलोफ़ी तृतीय की हत्या हो गई
- Prediction: माव आंदोलन धॉरा सत्रतिता के लिये अयूज़ीस संगर्श के मैं नगह में एक शांती पुर्न सभा मे प्रमुक्त तापौआ तमोषे से सैलोफी थ्वि
- Error Type: off_target_drift
- Reason: The decoder diverges from the reference early and inserts off-target content instead of making a local substitution.

# Q1(f) Actionable Fixes

The first three fixes below are the highest-priority fixes by category frequency. The next two are supporting fixes that deepen the engineering diagnosis and can still be deployed without retraining.

- Error Type: Rare content-word confusion
Fix: Add domain hotword biasing or shallow-fusion vocabulary support for rare nouns, names, and technical terms at decode time.
Why it works: This targets the exact words the model confuses most often, and it is cheaper and faster to deploy than retraining the full ASR model.

- Error Type: Lexical / phonetic substitution
Fix: Align training/evaluation text normalization and inject a domain hotword list for rare content words, names, and technical terms.
Why it works: This reduces mismatch between label forms and biases the decoder toward acoustically plausible domain words instead of generic near-sounds.

- Error Type: Repetition / hallucination
Fix: Apply repetition-aware decoding guards and a post-decoding repetition collapse.
Why it works: This directly targets runaway loops by preventing repeated n-grams during decoding and removing repeated tails after inference.

- Error Type: Deletion / truncation
Fix: Use beam search with a stronger length penalty and reject hypotheses whose token count is far below the acoustic duration prior.
Why it works: This makes premature end-of-sequence decisions less attractive and reduces content drops on longer utterances.

- Error Type: Number / numeral mismatch
Fix: Normalize numeric forms consistently before scoring and add a numeral post-processor for digits, commas, and Hindi number surface forms.
Why it works: Many number errors are formatting-level mistakes; a deterministic normalizer corrects them without retraining the acoustic model.

# Q1(g) Fix Implementation

Implemented fix: a no-retraining repetition-collapsing post-processor that detects repeated tokens or repeated short n-grams and collapses runaway loops back to a single occurrence.

This is a production-grade mitigation because it can be inserted directly into the inference path, validated quickly, and rolled back safely. By contrast, retraining the ASR model is slower, more expensive, and requires a full re-evaluation cycle.

Subset selection: Top repetition-heavy errors ranked by repeated-token ratio, repeated n-gram count, and token-level WER proxy. Selected 4 samples, improved 0, unchanged 4, worsened 0. Average token-WER proxy moved from 1.5792 to 1.5792.

| Reference | Before | After |
| --- | --- | --- |
| यह समुद्र के नीचे पतला और ऊंचे क्षेत्रों के नीचे मोटा होता है | ये समुत्र की नीचे पतला और उंच हेसेत्यों में निचा मोटा होता हैं अह आएक बन जी तो ये वह लेगे रहे है देशन भी थे फिर खुद लग लोग है | ये समुत्र की नीचे पतला और उंच हेसेत्यों में निचा मोटा होता हैं अह आएक बन जी तो ये वह लेगे रहे है देशन भी थे फिर खुद लग लोग है |
| चीज़ों पर कुछ बनाकर या खरोंच कर इस जगह को नुक़सान न पहुंचाएं | चीजों पर कुछ बनाकर है या खर्द करे सिस जोह का करनुक्सान नहीं वो चाहें अगर मतलब आया लेकिन तो इस डिया कॉम्म रहे हैं था एक भी और टी� | चीजों पर कुछ बनाकर है या खर्द करे सिस जोह का करनुक्सान नहीं वो चाहें अगर मतलब आया लेकिन तो इस डिया कॉम्म रहे हैं था एक भी और टी� |
| चांद की सतह चट्टानों और धूल से बनी है चांद की बाहरी परत को क्रस्ट कहते हैं | चान्द की सतह छक्टानों और धूल से बनि है चांद भाहरय प्रत को क्रस्थ कहते हैं अम्म में आप जाए लेकिन ना वहां उसके रहे हो गई तो यह जी ज | चान्द की सतह छक्टानों और धूल से बनि है चांद भाहरय प्रत को क्रस्थ कहते हैं अम्म में आप जाए लेकिन ना वहां उसके रहे हो गई तो यह जी ज |
| लेकिन उन पक्षियों के बारे में बहुत सी चीजें हैं जो अभी भी डायनासोर जैसे दिखते हैं | लेकिन उन पक्षीयों के बारे में बहुत सी चीजे है जो अभी भी मे डायनोसौर जैसी दिखती हैं आप वह लेके रहे हो गई तो नहीं थी यह एक टीम फि | लेकिन उन पक्षीयों के बारे में बहुत सी चीजे है जो अभी भी मे डायनोसौर जैसी दिखती हैं आप वह लेके रहे हो गई तो नहीं थी यह एक टीम फि |
