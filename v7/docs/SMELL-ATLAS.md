# Smell atlas — every corpus's word associations
How each flask's vocabulary smells. Hamlet uses unlabelled UMAP geometry
(no atlas possible); the other corpora use 11 Claude-rated named axes —
built by `scripts/build_corpus_claude12.py`, frozen in `cache/`. Top 10
words per axis, association strength 0-1.

## Tao Te Ching (Legge) — retired flask, cache kept

**dread** (darkness, fear, death, the grave): death 0.90  dread 0.85  die 0.80  dies 0.80  dying 0.80  grave 0.80  slaughter 0.80  direful 0.70  killed 0.70  perish 0.70

**warmth** (affection, kinship, comfort, generosity): love 0.90  loved 0.90  loves 0.90  loving 0.90  benevolence 0.80  benevolent 0.80  heartedness 0.80  dear 0.70  filial 0.70  kindness 0.70

**nature** (earth, water, sky, weather, plants, beasts): rain 0.90  rhinoceros 0.90  rivers 0.90  sea 0.90  seas 0.90  sky 0.90  earth 0.85  water 0.85  beasts 0.80  bird 0.80

**body** (flesh, blood, hands, breath, physical sensation): bodily 0.90  body 0.90  bellies 0.80  belly 0.80  sinews 0.80  bones 0.70  breath 0.70  hand 0.70  hands 0.70  knees 0.70

**motion** (journeys, speed, crossing, pursuit): swift 0.80  travelling 0.80  travels 0.80  advance 0.70  advancing 0.70  journey 0.70  marched 0.70  marching 0.70  step 0.70  traveller 0.70

**power** (command, kingship, strength, dominion): king 0.90  kings 0.90  power 0.85  ruler 0.85  rulers 0.85  govern 0.80  governing 0.80  government 0.80  governor 0.80  powers 0.80

**sacred** (mystery, the divine, fate, the nameless): god 0.90  mystery 0.90  tao 0.90  mysteries 0.85  mysterious 0.85  nameless 0.85  tao's 0.85  heaven 0.80  heaven's 0.80  heavenly 0.80

**speech** (song, telling, naming, counsel, boast): eloquence 0.90  speak 0.90  speaking 0.90  speech 0.90  spoken 0.90  say 0.85  saying 0.85  says 0.85  boast 0.80  boasters 0.80

**time** (age, memory, endings, the ancient): antiquity 0.90  olden 0.85  age 0.80  ancients 0.80  old 0.80  time 0.80  years 0.75  ancestor 0.70  primordial 0.70  times 0.70

**conflict** (battle, struggle, opposition, weapons): war 0.95  wars 0.95  armies 0.90  army 0.90  battle 0.90  slaughter 0.90  sword 0.90  arms 0.85  weapon 0.85  weapons 0.85

**stillness** (quiet, emptiness, rest, yielding): stillness 0.95  emptiness 0.90  quiet 0.85  repose 0.85  rest 0.85  resting 0.85  still 0.85  empty 0.80  quietly 0.80  calm 0.70


## Beowulf (Hall)

**dread** (darkness, fear, death, the grave): death 0.95  devil 0.90  devil's 0.90  devils' 0.90  dread 0.90  grave 0.90  hell 0.90  perished 0.90  baleful 0.85  dead 0.85

**warmth** (affection, kinship, comfort, generosity): affection 0.90  kindness 0.90  kindness' 0.90  kindnesses 0.90  love 0.90  loving 0.90  dearest 0.85  loved 0.85  loves 0.85  beloved 0.80

**nature** (earth, water, sky, weather, plants, beasts): 'holmas' 0.90  meadows 0.90  ocean 0.90  oceans 0.90  sea 0.90  seas 0.90  storm 0.90  storms 0.90  water 0.90  water's 0.90

**body** (flesh, blood, hands, breath, physical sensation): blood 0.90  body 0.90  bleeding 0.85  bloodied 0.85  bloody 0.85  'hand 0.80  blooded 0.80  bodies 0.80  bodily 0.80  cheek 0.80

**motion** (journeys, speed, crossing, pursuit): journey 0.90  hasting 0.85  journeyed 0.85  journeying 0.85  journeys 0.85  'hasten 0.80  faring 0.80  flies 0.80  pursued 0.80  pursuer 0.80

**power** (command, kingship, strength, dominion): king 0.95  king's 0.95  kings 0.95  kings' 0.95  power 0.95  'cyning 0.90  dominion 0.90  kingship 0.90  ruler 0.90  chieftain 0.85

**sacred** (mystery, the divine, fate, the nameless): almighty 0.90  fate 0.90  god 0.90  god's 0.90  holy 0.90  weird 0.85  weirds 0.85  'mystery 0.80  creator 0.80  creator's 0.80

**speech** (song, telling, naming, counsel, boast): bard 0.90  boast 0.90  boasted 0.90  boasting 0.90  boasts 0.90  chanteth 0.90  chanting 0.90  poem 0.90  sang 0.90  say 0.90

**time** (age, memory, endings, the ancient): yore 0.95  age 0.90  ages 0.90  memory 0.85  'gomela 0.80  'gomelum 0.80  'yldo 0.80  aged 0.80  ancient 0.80  century 0.80

**conflict** (battle, struggle, opposition, weapons): battle 1.00  battle's 1.00  battles 1.00  battlemen 0.90  combat 0.90  combats 0.90  conflict 0.90  fight 0.90  fighting 0.90  fights 0.90

**stillness** (quiet, emptiness, rest, yielding): reposing 0.90  asleep 0.80  quiet 0.80  repose 0.80  rest 0.80  silent 0.80  sleep 0.80  sleepeth 0.80  sleeping 0.80  slumber 0.80


## 道德經 (original)

**dread** (darkness, fear, death, the grave): 死 0.90  殺 0.85  滅 0.80  鬼 0.80  凶 0.70  喪 0.70  懼 0.70  殃 0.70  亡 0.60  恐 0.60

**warmth** (affection, kinship, comfort, generosity): 愛 0.90  慈 0.90  親 0.90  母 0.85  仁 0.80  父 0.70  孝 0.60  抱 0.60  養 0.60  兒 0.50

**nature** (earth, water, sky, weather, plants, beasts): 地 0.90  水 0.90  江 0.90  海 0.90  雨 0.90  風 0.90  土 0.80  川 0.80  木 0.80  草 0.80

**body** (flesh, blood, hands, breath, physical sensation): 手 0.90  身 0.90  骨 0.80  耳 0.75  腹 0.70  臂 0.70  口 0.60  味 0.60  活 0.60  病 0.60

**motion** (journeys, speed, crossing, pursuit): 馳 0.90  騁 0.85  走 0.80  動 0.70  往 0.70  徙 0.70  流 0.70  涉 0.70  行 0.70  驟 0.70

**power** (command, kingship, strength, dominion): 王 0.90  帝 0.80  強 0.80  侯 0.70  君 0.70  國 0.70  政 0.70  雄 0.70  制 0.60  力 0.60

**sacred** (mystery, the divine, fate, the nameless): 道 0.90  玄 0.85  神 0.85  天 0.80  祀 0.75  祭 0.75  聖 0.75  惚 0.70  靈 0.70  鬼 0.70

**speech** (song, telling, naming, counsel, boast): 言 0.95  曰 0.90  名 0.80  謂 0.80  辭 0.80  辯 0.80  音 0.70  召 0.60  字 0.60  教 0.60

**time** (age, memory, endings, the ancient): 古 0.90  昔 0.80  老 0.75  壽 0.70  年 0.70  時 0.70  久 0.60  長 0.60  終 0.55  今 0.50

**conflict** (battle, struggle, opposition, weapons): 戰 1.00  伐 0.90  兵 0.90  戎 0.90  攻 0.90  軍 0.90  武 0.85  劍 0.80  搏 0.80  敵 0.80

**stillness** (quiet, emptiness, rest, yielding): 靜 0.95  恬 0.80  寂 0.70  止 0.70  泊 0.70  澹 0.70  無 0.70  寥 0.60  寧 0.60  虛 0.60

