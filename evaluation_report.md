# XSum BART Evaluation Report

## Context
- Model checkpoint: `models/bart_xsum_finetuned`
- Dataset split: `test`
- Test examples: 11334
- Device: `cuda`
- Generation: 4 beams, max length 64
- Checkpoint note: this is the 1-epoch fine-tuned `facebook/bart-base` model.
- Article excerpt note: decoded article excerpts may already be truncated to 512 BART tokens by design from preprocessing.
- Summary length plot: `outputs/evaluation/bart_xsum_finetuned/summary_length_distribution.png`

## Metrics
ROUGE is computed with `use_stemmer=True`.

| Metric | Value |
| --- | ---: |
| ROUGE-1 | 0.3938 |
| ROUGE-2 | 0.1696 |
| ROUGE-L | 0.3197 |
| ROUGE-Lsum | 0.3196 |
| BERTScore precision mean | 0.9136 |
| BERTScore precision std | 0.0301 |
| BERTScore recall mean | 0.9 |
| BERTScore recall std | 0.0298 |
| BERTScore F1 mean | 0.9066 |
| BERTScore F1 std | 0.0282 |
| Reference summary length mean | 26.12 |
| Generated summary length mean | 21.39 |

## Baseline Comparison
Published BART-large XSum reference scores are ROUGE-1 45.14, ROUGE-2 22.27, and ROUGE-L 37.25.
This is a reference point, not an exact apples-to-apples comparison, because this project uses `facebook/bart-base` and the current checkpoint was fine-tuned for 1 epoch.

Source: https://paperswithcode.com/paper/bart-denoising-sequence-to-sequence-pre

## Observations
- The model usually follows the XSum one-sentence summary format.
- The 1-epoch BART-base checkpoint trails the published BART-large reference baseline, which is expected.
- Generated summary length is close to the reference length distribution.

## Strongest Examples
### Example 11042 - ROUGE-L 1.0
Article excerpt: A selection of your pictures of Scotland sent in between 3 January and 10 February. Send your photos to scotlandpictures@bbc.co.uk or our Instagram at #bbcscotlandpics

Reference: All pictures are copyrighted.

Generated: All pictures are copyrighted.

### Example 8775 - ROUGE-L 1.0
Article excerpt: A selection of your pictures of Scotland sent in between 26 May and 2 June. Send your photos to scotlandpictures@bbc.co.uk or via Instagram at #bbcscotlandpics

Reference: All pictures are copyrighted.

Generated: All pictures are copyrighted.

### Example 8562 - ROUGE-L 1.0
Article excerpt: A selection of your pictures of Scotland sent in between 6 and 13 January. Send your photos to scotlandpictures@bbc.co.uk or our Instagram at #bbcscotlandpics.

Reference: All pictures are copyrighted.

Generated: All pictures are copyrighted.

### Example 6473 - ROUGE-L 1.0
Article excerpt: 1453 - Sultan Mehmed II the Magnificent captures Constantinople, ending Byzantine Empire and consolidating Ottoman Empire in Asia Minor and Balkans.
15th-16th centuries - Expansion into Asia and Africa.
1683 - Ottoman advance into Europe halted at Battle of Vienna. Long decline begins.
19th century - Efforts at political and economic modernisation of Empire largely founder.
1908 - Young Turk Revolution establishes constitutional rule, but degenerates into military dictatorship during First World War, where Ottoman Empire fights in alliance with Germany and Austria-Hungary.
1918-22 - Partition of defeated Ottoman Empire leads to eventual triumph of Turkish National Movement in war of independence against foreign occupation and rule of Sultan

Reference: A chronology of key events:

Generated: A chronology of key events:

### Example 6372 - ROUGE-L 1.0
Article excerpt: Find out how you can join in and submit your images and videos below.
If you have a picture you'd like to share, email us at england@bbc.co.uk, post it on Facebook or tweet it to @BBCEngland. You can also find us on Instagram - use #englandsbigpicture to share an image there. You can also see a recent archive of pictures on our England's Big Picture board on Pinterest.
When emailing pictures, please make sure you include the following information:
Please note that whilst we welcome all your pictures, we are more likely to use those which have been taken in the past week.
If you submit a picture, you do so in accordance with the BBC's Terms and Conditions.
In contributing to England's Big Picture you agree to grant us a royalty-free, non-exc

Reference: Each day we feature a photograph sent in from across England.

Generated: Each day we feature a photograph sent in from across England.

## Weakest Examples
### Example 24 - ROUGE-L 0.0
Article excerpt: Two snowsports enthusiasts got married at a Scottish ski resort before sliding off down a run in their wedding attire.
Bridget and Jonathan Reid, from Moy, near Tomatin in the Highlands, tied the knot at Nevis Range, near Fort William, on Friday.
The couple first's date six years ago was a skiing trip, so they decided it would be appropriate to get married on skis.
Adventure photographer Hamish Frost took their wedding snaps.
Bridget, who is a teacher, and Jonathan, who runs his own electrical automation company, benefited from recent snowfalls for their big day.
They got married in full Highland dress, which includes a kilt, and white wedding dress surrounded by snow-covered mountain landscape.
The white stuff had been lacking over winter,

Reference: All images copyrighted.

Generated: A couple have been married on skis for the first time.

### Example 322 - ROUGE-L 0.0
Article excerpt: That's according to a new report by a senior group of MPs.
Parliament's Intelligence and Security Committee said recruiters should try websites like Mumsnet to help increase the proportion of female spies.
It wants more of them working in places like MI5, MI6 and communications spy centre GCHQ.
The report says women in the intelligence services are being held back by a layer of male, middle managers labelled "the permafrost" who have a "very traditional male mentality and outlook".
Mumsnet chief executive Justine Roberts responded to the call for recruiters to use things like her website but we're thinking she wasn't being entirely serious.
"I'm afraid I'm unable to comment as I have an urgent appointment with a rock in St. James's Park."
W

Reference: Forget James Bond, when it comes to recruiting spies needed to protect Britain there aren't enough Jane Bonds.

Generated: Women in the intelligence services are being held back by "the permafrost" who have a "very traditional male mentality and outlook".

### Example 960 - ROUGE-L 0.0
Article excerpt: The way we work, play and live with robots is changing.
In a special series Ricky travels the country meeting the robots of the future and the scientists working on them.
From spending a night in a robot house to getting a brain scan, Ricky finds out how and why our relationship with robots is changing, fast.
Check out his first report here...

Reference: They can walk, they can talk, and may soon be thinking for themselves.

Generated: Meet the robots of the future.

### Example 1314 - ROUGE-L 0.0
Article excerpt: The Stanford University analysis of 68 million days' worth of minute-by-minute data showed the average number of daily steps was 4,961.
Hong Kong was top averaging 6,880 a day, while Indonesia was bottom of the rankings with just 3,513.
But the findings also uncovered intriguing details that could help tackle obesity.
Most smartphones have a built-in accelerometer that can record steps and the researchers used anonymous data from more than 700,000 people who used the Argus activity monitoring app.
Scott Delp, a professor of bioengineering and one of the researchers, said: "The study is 1,000 times larger than any previous study on human movement.
"There have been wonderful health surveys done, but our new study provides data from more count

Reference: US scientists have amassed "planetary-scale" data from people's smartphones to see how active we really are.

Generated: The average number of steps a day in a country is 1,000 times larger than any previous study on human movement, a study suggests.

### Example 1670 - ROUGE-L 0.0
Article excerpt: The mosaics, part of a 1,700-year-old town house, were found on the site of a development in Leicester.
Open days attracted thousands of people but a question mark hung over the fate over the most spectacular finds.
Now city mayor Sir Peter Soulsby has confirmed they will form part of a Â£7m revamp of the nearby Jewry Wall museum.
The excavation recovered hundreds of artefacts - including a gruesome carved handle depicting people being fed to lions - but the mosaics were the largest found in the city for 150 years.
Speaking at the final open day on Sunday, archaeologist Jon Coward said: "The big mosaic will be lifted and conserved but quite where it will end up I couldn't tell you but it should be made available.
"One of the things that cou

Reference: High-status Roman floors discovered by archaeologists during building work will go on permanent display, it has been confirmed.

Generated: One of the largest mosaics ever found in the UK is to be made available to the public.
