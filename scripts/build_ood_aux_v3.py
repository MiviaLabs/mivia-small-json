#!/usr/bin/env python3
"""Build corpus v3 (multilingual owner decision, 2026-10-10) and the
German in-domain ticket file.

Language never implies out-of-scope (owner: "everything must be
multilingual — that was the whole point"). Two artifacts:

1. ``data/real/ood-aux/ood-aux-train-v3.jsonl`` — corpus v2 MINUS the
   ticket-non-en category (2,000 rows that taught language→OOD) PLUS
   agent-authored multilingual free-chat (content-based OOD across the
   served languages, so the OOD class does not decay into English-only).
2. ``data/real/multilang-tickets/multilang-de-tickets.jsonl`` — 1,500
   real German incident tickets as ``issue`` rows (the one non-EN
   language with real ticket data in the repo: the 20k CSV is de 8,077 /
   en 11,923), hash-disjoint from the tickets-de retention slice (the
   CSV's first 300 qualifying de rows, recomputed identically to
   scripts/eval_ood_class.py) and from every eval surface / corpus file.

Disjointness is verified by normalized-text hash; both artifacts are
sha-pinned by tests/test_ood_aux_disjoint.py.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_ood_aux_hf import _norm, banned_texts  # noqa: E402

AUX_DIR = ROOT / "data/real/ood-aux"
V2_CORPUS = AUX_DIR / "ood-aux-train-v2.jsonl"
V3_CORPUS = AUX_DIR / "ood-aux-train-v3.jsonl"
TICKETS_DIR = ROOT / "data/real/multilang-tickets"
TICKETS_OUT = TICKETS_DIR / "multilang-de-tickets.jsonl"
CSV_PATH = (ROOT / "data/incoming/external/incident-severity"
            / "tickets-multi-lang-20k.csv")
N_TICKETS = 1500
SEED = 20261011

# Agent-authored multilingual free-chat (owner annotation directive: the
# session LLM authors; no templates; varied registers — greetings, small
# talk, personal news, thanks, jokes, plans). 40 rows x 8 languages, the
# languages of the v2 eval's non-EN slices and the served severity heads.
MULTILINGUAL_FREE_CHAT: dict[str, list[str]] = {
    "de": [
        "Na, wie lief dein erster Arbeitstag? Alles gut gelaufen?",
        "Ich hab das neue Restaurant in der Innenstadt probiert, echt lecker.",
        "Kommst du Samstag zum Fußballgucken? Wir treffen uns um acht.",
        "Das Wetter ist ja heute wieder herrlich, ich geh nachher spazieren.",
        "Hast du die Folge gestern Abend gesehen? Ich hab geweint vor Lachen.",
        "Vielen Dank nochmal für deine Hilfe letzte Woche!",
        "Mein Zug hatte heute wieder 40 Minuten Verspätung, unglaublich.",
        "Sollen wir am Wochenende wandern gehen? Das Wetter soll nice werden.",
        "Ich habe endlich mein Fahrrad repariert, nach drei Monaten.",
        "Grüß dich! Lange nicht gesehen, wie geht es dir?",
        "Die Kinder sind im Bett, endlich Feierabend für uns.",
        "Ich überlege, mir ein Konzertticket zu Weihnachten zu schenken.",
        "Kannst du mir dein Rezept für den Apfelkuchen schicken?",
        "Der Film war ganz okay, aber das Ende war irgendwie unbefriedigend.",
        "Beim Zahnarzt war ich heute, das war so unangenehm wie immer.",
        "Wir haben gestern Karten gespielt, ich habe grandios verloren.",
        "Wie war dein Urlaub? Die Bilder sahen fantastisch aus!",
        "Ich lerne gerade Klavier, es klingt noch furchtbar.",
        "Sag mal, hast du Lust auf ein Spiel Billiard am Donnerstag?",
        "Meine Oma hat wieder Kekse geschickt, ein ganzes Paket voll.",
        "Der Hund hat heute den Briefträger angebellt, wie immer.",
        "Ich war im Schwimmbad, die Bahn war heute herrlich leer.",
        "Kaffee geht immer, sag einfach wann.",
        "Hab die Prüfung bestanden! Ich bin so erleichtert.",
        "Es schüttet seit Stunden, ich bin ganz nass geworden.",
        "Wollen wir danach noch ein Eis essen gehen?",
        "Ich habe verschlafen und der Wecker hat nicht geklingelt.",
        "Gute Fahrt morgen! Melde dich, wenn du angekommen bist.",
        "Der Weihnachtsmarkt öffnet diese Woche, lass uns hingehen.",
        "Mein Handyakku hält nicht mehr einen Tag, Zeit für ein neues.",
        "Warst du schon mal in der neuen Bibliothek? Riesig!",
        "Ich habe gestern komplett den Fitnessstudio-Termin verpasst.",
        "Alles Gute zum Geburtstag! Feier schön heute Abend.",
        "Der Nachbar baut schon wieder um, der Bohrer läuft nonstop.",
        "Hast du Tipps für Serien? Mir geht langsam nichts mehr aus.",
        "Beim Sport habe ich mir aufgehört den Rücken zu verrennen.",
        "Wir haben ein neues Aquarium, die Fische heißen Salz und Pfeffer.",
        "Müde bin ich, aber schlafen kann ich einfach nicht.",
        "Der Kuchen fürs Büro ist schon fertig, ich freu mich auf die Gesichter.",
        "Gehen wir diese Woche einmal schwimmen nach der Arbeit?",
    ],
    "fr": [
        "Alors, c'était bien ton premier jour de boulot ?",
        "J'ai testé le nouveau resto du centre-ville, franchement délicieux.",
        "Tu viens regarder le match samedi ? On se retrouve à huit heures.",
        "Il fait super beau aujourd'hui, je vais aller marcher un peu.",
        "T'as vu l'épisode d'hier soir ? J'ai pleuré de rire.",
        "Merci encore pour ton coup de main la semaine dernière !",
        "Mon train avait encore quarante minutes de retard, incroyable.",
        "On randonne ce week-end ? La météo annonce du beau temps.",
        "J'ai enfin réparé mon vélo, après trois mois.",
        "Salut ! Ça fait longtemps, comment vas-tu ?",
        "Les enfants sont couchés, enfin un peu de tranquillité.",
        "Je pense m'offrir un billet de concert pour Noël.",
        "Tu peux m'envoyer ta recette du gâteau aux pommes ?",
        "Le film était correct, mais la fin m'a laissé sur ma faim.",
        "Je suis allé chez le dentiste, aussi désagréable que d'habitude.",
        "On a joué aux cartes hier, j'ai magnifiquement perdu.",
        "Ça s'est passé tes vacances ? Les photos avaient l'air top !",
        "J'apprends le piano, ça sonne encore affreusement.",
        "Ça te dit une partie de billard jeudi ?",
        "Ma grand-mère a encore envoyé des biscuits, un carton entier.",
        "Le chien a encore aboyé sur le facteur, comme toujours.",
        "J'ai été à la piscine, les couloirs étaient agréables vides.",
        "Un café, ça va toujours, dis-moi juste quand.",
        "J'ai eu mon examen ! Je suis tellement soulagé.",
        "Il pleut àverse depuis des heures, je suis trempé.",
        "On prend une glace après ?",
        "J'ai dormi trop tard, le réveil n'a pas sonné.",
        "Bon voyage demain ! Écris-moi quand tu arrives.",
        "Le marché de Noël ouvre cette semaine, allons-y.",
        "La batterie de mon téléphone ne tient plus une journée.",
        "T'es déjà allé à la nouvelle bibliothèque ? Immense !",
        "J'ai encore raté ma séance de sport, trop de boulot.",
        "Joyeux anniversaire ! Amuse-toi bien ce soir.",
        "Le voisin refait des travaux, la perceuse tourne sans arrêt.",
        "T'as des séries à me conseiller ? Je n'ai plus rien à voir.",
        "On devrait aller nager cette semaine après le travail.",
        "On a un nouvel aquarium, les poissons s'appellent Sel et Poivre.",
        "Je suis crevé, mais impossible de dormir.",
        "Le gâteau pour le bureau est prêt, j'ai hâte de voir leurs têtes.",
        "Rien de tel qu'un bon thé par ce temps gris.",
    ],
    "es": [
        "¿Qué tal tu primer día de trabajo? ¿Todo bien?",
        "Probé el restaurante nuevo del centro, riquísimo.",
        "¿Vienes a ver el partido el sábado? Quedamos a las ocho.",
        "Qué día tan bonito hoy, me voy a caminar más tarde.",
        "¿Viste el episodio de anoche? Me morí de risa.",
        "¡Gracias otra vez por tu ayuda la semana pasada!",
        "Mi tren tuvo otra vez cuarenta minutos de retraso, increíble.",
        "¿Hacemos una ruta el fin de semana? Dicen que hará bueno.",
        "Por fin arreglé la bici, después de tres meses.",
        "¡Hola! Cuánto tiempo, ¿cómo estás?",
        "Los niños ya duermen, por fin tranquilidad.",
        "Estoy pensando regalarme una entrada de concierto por Navidad.",
        "¿Me pasas tu receta de la tarta de manzana?",
        "La película estuvo bien, pero el final me dejó frío.",
        "Fui al dentista, tan desagradable como siempre.",
        "Ayer jugamos a las cartas, perdí de manera espectacular.",
        "¿Qué tal las vacaciones? ¡Las fotos tenían una pinta genial!",
        "Estoy aprendiendo piano, suena fatal todavía.",
        "¿Te apetece una partida de billar el jueves?",
        "Mi abuela mandó otra vez galletas, una caja entera.",
        "El perro le ladró otra vez al cartero, como siempre.",
        "Fui a la piscina, los carriles bastante vacíos.",
        "Un café siempre apetece, dime cuándo.",
        "¡Aprobé el examen! Qué alivio.",
        "Lleva horas lloviendo, me empapé entera.",
        "¿Nos tomamos un helado después?",
        "Me quedé dormido y el despertador no sonó.",
        "¡Buen viaje mañana! Escríbeme cuando llegues.",
        "El mercadillo de Navidad abre esta semana, vamos.",
        "La batería del móvil ya no dura ni un día.",
        "¿Has estado ya en la biblioteca nueva? ¡Enorme!",
        "Volví a perderme mi sesión de gimnasio por el trabajo.",
        "¡Feliz cumpleaños! Diviértete esta noche.",
        "El vecino vuelve a hacer obras, el taladro no para.",
        "¿Me recomiendas series? Ya no sé qué ver.",
        "Deberíamos ir a nadar esta semana tras el trabajo.",
        "Tenemos un acuario nuevo, los peces se llaman Sal y Pimienta.",
        "Estoy agotado, pero no me duermo.",
        "El pastel para la oficina ya está, quiero ver sus caras.",
        "Un calentito en este día gris no me lo quita nadie.",
    ],
    "pl": [
        "No i jak pierwszy dzień w pracy? Wszystko ok?",
        "Próbowałem nowej restauracji w centrum, naprawdę smaczne.",
        "Idziesz w sobotę na mecz? Spotykamy się o ósmej.",
        "Pogoda dziś prześliczna, idę później na spacer.",
        "Widziałeś wczoraj odcinek? Płakałem ze śmiechu.",
        "Dzięki jeszcze raz za pomoc w zeszłym tygodniu!",
        "Mój pociąg znowu miał czterdzieści minut opóźnienia, niewiarygodne.",
        "Idziemy w góry w weekend? Ma być ładnie.",
        "W końcu naprawiłem rower, po trzech miesiącach.",
        "Hej! Dawno cię nie widziałem, jak leci?",
        "Dzieci już śpią, w końcu chwila spokoju.",
        "Myślę, żeby podarować sobie bilet na koncert na święta.",
        "Podasz mi przepis na szarlotkę?",
        "Film był w porządku, ale zakończenie jakoś niesatysfakcjonujące.",
        "Byłem dziś u dentysty, równie nieprzyjemnie jak zawsze.",
        "Wczoraj graliśmy w karty, przegrałem widowiskowo.",
        "Jak było na wakacjach? Zdjęcia wyglądały fantastycznie!",
        "Uczę się gry na pianinie, brzmi jeszcze okropnie.",
        "Masz ochotę na bilarda w czwartek?",
        "Babcia znowu przysłała ciasteczka, całe pudło.",
        "Pies zaszczekał dziś na listonosza, jak zwykle.",
        "Byłem na basenie, tory dziś przyjemnie puste.",
        "Kawa zawsze dobra, powiedz tylko kiedy.",
        "Zdałem egzamin! Ogromna ulga.",
        "Leje od godzin, przemokłem na wskroś.",
        "Zjemy po wszystkim loda?",
        "Zaspałem i budzik nie zadzwonił.",
        "Szczęśliwej drogi jutro! Napisz, jak dotrzesz.",
        "Jarmark bożonarodzeniowy otwiera się w tym tygodniu, chodźmy.",
        "Bateria w telefonie nie wytrzymuje już dnia.",
        "Byłeś już w nowej bibliotece? Ogromna!",
        "Znowu przegapiłem trening przez robotę.",
        "Wszystkiego najlepszego z okazji urodzin! Baw się dziś dobrze.",
        "Sąsiad znowu remontuje, wiertarka chodzi bez przerwy.",
        "Polecisz jakieś seriale? Już nie wiem, co oglądać.",
        "Powinniśmy iść w tym tygodniu popływać po pracy.",
        "Mamy nowe akwarium, rybki nazywają się Sól i Pieprz.",
        "Jestem zmęczony, ale nie mogę zasnąć.",
        "Ciasto do biura już gotowe, nie mogę się doczekać ich reakcji.",
        "Nie ma nic lepszego niż grzaniec w taki szary dzień.",
    ],
    "pt": [
        "E então, como foi o teu primeiro dia de trabalho?",
        "Experimentei o restaurante novo no centro, uma delícia.",
        "Vens ver o jogo ao sábado? Encontramo-nos às oito.",
        "O tempo hoje está lindo, vou dar um passeio logo.",
        "Viste o episódio de ontem à noite? Ri-me às lágrimas.",
        "Mais uma vez obrigado pela ajuda da semana passada!",
        "O meu comboio teve mais uma vez quarenta minutos de atraso.",
        "Fazemos uma caminhada no fim de semana? Vai estar bom.",
        "Finalmente consertei a bicicleta, depois de três meses.",
        "Olá! Quanto tempo, como estás?",
        "As crianças já estão a dormir, finalmente calma.",
        "Estou a pensar oferecer-me um bilhete de concertos no Natal.",
        "Podes mandar-me a tua receita de bolo de maçã?",
        "O filme até foi giro, mas o final deixou-me a desejar.",
        "Fui ao dentista, tão desagradável como sempre.",
        "Ontem jogámos cartas, perdi de forma espectacular.",
        "Correram bem as férias? As fotos pareciam fantásticas!",
        "Estou a aprender piano, ainda soa horrível.",
        "Apetece-te uma partida de snooker na quinta?",
        "A avó voltou a enviar biscoitos, uma caixa inteira.",
        "O cão latiu outra vez ao carteiro, como sempre.",
        "Fui à piscina, as pistas agradavelmente vazias.",
        "Um café cai sempre bem, diz só quando.",
        "Passei no exame! Que alívio.",
        "Está a chover há horas, fiquei encharcado.",
        "Tomamos um gelado depois?",
        "Adormeci e o despertador não tocou.",
        "Boa viagem amanhã! Escreve quando chegares.",
        "O mercado de Natal abre esta semana, vamos.",
        "A bateria do telemóvel já não dura um dia.",
        "Já foste à biblioteca nova? Enorme!",
        "Faltei outra vez ao ginásio por causa do trabalho.",
        "Parabéns pelo aniversário! Diverte-te hoje à noite.",
        "O vizinho está outra vez a fazer obras, a broca não para.",
        "Tens séries para me recomendar? Já não sei o que ver.",
        "Devíamos ir nadar esta semana depois do trabalho.",
        "Temos um aquário novo, os peixes chamam-se Sal e Pimenta.",
        "Estou exausto, mas não consigo dormir.",
        "O bolo para o escritório já está, mal posso esperar pelas caras.",
        "Nada como um chá quente num dia cinzento destes.",
    ],
    "it": [
        "Allora, com'è andato il primo giorno di lavoro?",
        "Ho provato il ristorante nuovo in centro, buonissimo.",
        "Vieni a vedere la partita sabato? Ci vediamo alle otto.",
        "Che bella giornata oggi, più tardi vado a fare due passi.",
        "Hai visto l'episodio di ieri sera? Sono morto dalle risate.",
        "Grazie ancora per l'aiuto la settimana scorsa!",
        "Il mio treno aveva di nuovo quaranta minuti di ritardo.",
        "Andiamo a camminare nel weekend? Dovrebbe essere bello.",
        "Ho finalmente riparato la bici, dopo tre mesi.",
        "Ciao! Quanto tempo, come stai?",
        "I bambini dormono, finalmente un po' di pace.",
        "Sto pensando di regalarmi un biglietto per un concerto a Natale.",
        "Mi mandi la tua ricetta della torta di mele?",
        "Il film era ok, ma il finale mi ha lasciato perplesso.",
        "Sono andato dal dentista, sgradevole come sempre.",
        "Ieri abbiamo giocato a carte, ho perso in modo spettacolare.",
        "Com'è andata la vacanza? Le foto sembravano fantastiche!",
        "Sto imparando il pianoforte, suona ancora da cani.",
        "Ti va una partita a biliardo giovedì?",
        "La nonna ha mandato di nuovo biscotti, una scatola intera.",
        "Il cane ha abbaiato al postino, come al solito.",
        "Sono stato in piscina, le corsie piacevolmente vuote.",
        "Un caffè va sempre bene, dimmi solo quando.",
        "Ho passato l'esame! Che sollievo.",
        "Piove da ore, mi sono bagnato fradicio.",
        "Ci prendiamo un gelato dopo?",
        "Ho dormito troppo e la sveglia non ha suonato.",
        "Buon viaggio domani! Scrivimi quando arrivi.",
        "Il mercatino di Natale apre questa settimana, andiamo.",
        "La batteria del telefono non dura più un giorno.",
        "Sei già stato in biblioteca nuova? Enorme!",
        "Ho saltato di nuovo la palestra per il lavoro.",
        "Buon compleanno! Divertiti stasera.",
        "Il vicino ristruttura di nuovo, il trapano non si ferma.",
        "Mi consigli delle serie? Non so più cosa guardare.",
        "Dovremmo andare a nuotare questa settimana dopo il lavoro.",
        "Abbiamo un acquario nuovo, i pesci si chiamano Sale e Pepe.",
        "Sono stanchissimo, ma non riesco a dormire.",
        "La torta per l'ufficio è pronta, non vedo l'ora delle facce.",
        "Niente di meglio di una tisana calda in una giornata grigia.",
    ],
    "nl": [
        "Nou, hoe was je eerste werkdag? Alles goed gegaan?",
        "Ik heb dat nieuwe restaurant in het centrum geprobeerd, echt lekker.",
        "Kom je zaterdag voetbal kijken? We treffen elkaar om acht.",
        "Het weer is vandaag weer heerlijk, ik ga straks wandelen.",
        "Heb je de aflevering gisteravond gezien? Ik huilde van het lachen.",
        "Nogmaal bedankt voor je hulp vorige week!",
        "Mijn trein had vandaag weer veertig minuten vertraging.",
        "Gaan we dit weekend wandelen? Het wordt mooi weer.",
        "Ik heb eindelijk mijn fiets gerepareerd, na drie maanden.",
        "Hoi! Lang niet gezien, hoe gaat het?",
        "De kinderen liggen op bed, eindelijk rust.",
        "Ik overweeg mezelf een concertkaartje te geven voor kerst.",
        "Kun je me je recept voor appeltaart sturen?",
        "De film was oké, maar het einde viel tegen.",
        "Ik was vandaag bij de tandarts, net zo vervelend als altijd.",
        "We speelden gisteren kaart, ik heb grandioos verloren.",
        "Hoe was je vakantie? De foto's zagen er fantastisch uit!",
        "Ik leer piano, het klinkt nog vreselijk.",
        "Zullen we donderdag een spelletje poolen?",
        "Mijn oma heeft weer koekjes gestuurd, een hele doos.",
        "De hond blafte vandaag weer naar de postbode, zoals altijd.",
        "Ik was in het zwembad, de banen waren lekker leeg.",
        "Koffie kan altijd, zeg maar wanneer.",
        "Ik heb het examen gehaald! Wat een opluchting.",
        "Het giert al uren van de regen, ik ben doorweekt.",
        "Gaan we daarna nog een ijsje eten?",
        "Ik heb me verslapen en de wekker ging niet af.",
        "Goede reis morgen! Laat iets horen als je aankomt.",
        "De kerstmarkt opent deze week, laten we gaan.",
        "Mijn telefoon houdt het geen dag meer vol.",
        "Ben je al in de nieuwe bibliotheek geweest? Enorm!",
        "Ik heb mijn sportschoolsessie weer gemist door werk.",
        "Gefeliciteerd met je verjaardag! Veel plezier vanavond.",
        "De buurman is weer aan het verbouwen, de boor draait non-stop.",
        "Heb je tips voor series? Ik weet niet meer wat ik moet kijken.",
        "We moeten deze week na het werk gaan zwemmen.",
        "We hebben een nieuw aquarium, de vissen heten Zout en Peper.",
        "Ik ben moe, maar ik kan gewoon niet slapen.",
        "De taart voor kantoor is klaar, ik kijk uit naar de gezichten.",
        "Niets beter dan een warme kop thee op zo'n grijze dag.",
    ],
    "ja": [
        "初日の仕事はどうだった？うまくいった？",
        "市内の新しいレストラン行ってみた、すごく美味しかった。",
        "土曜日のサッカー観戦くる？８時に集まるよ。",
        "今日もいい天気だね、あとで散歩に行く。",
        "昨日の夜のエピソード見た？笑いすぎて涙出た。",
        "先週は手伝ってくれて本当にありがとう！",
        "今日も電車が４０分遅れた、信じられない。",
        "週末ハイキング行かない？天気いいらしいよ。",
        "やっと自転車直した、３ヶ月ぶりに。",
        "やあ！久しぶり、元気にしてる？",
        "子供たち寝た、やっと自分の時間。",
        "クリスマスにコンサートのチケット自分に買おうかな。",
        "アップルパイのレシピ送ってもらえる？",
        "映画はまあまあだったけど、ラストがいまいちだった。",
        "今日は歯医者行った、いつも通り嫌だった。",
        "昨日トランプした、見事に負けた。",
        "休暇はどうだった？写真すごく良さそうだった！",
        "ピアノ習ってるんだけど、まだひどい音。",
        "木曜日ビリヤードどう？",
        "祖母がまたクッキー送ってきた、箱いっぱい。",
        "犬が今日も配達員に吠えた、いつものこと。",
        "プール行った、コースが空いてて気持ちよかった。",
        "コーヒーはいつでもいい、時間決めて。",
        "試験受かった！ほっとした。",
        "何時間も雨降ってて、ずぶ濡れになった。",
        "あとでアイス食べに行かない？",
        "寝坊した、目覚ましならなかった。",
        "明日、良い旅を！着いたら連絡して。",
        "クリスマスマーケット今週開くらしい、行こう。",
        "スマホの電池が１日もたなくなった。",
        "新しい図書館行った？めっちゃ大きいよ。",
        "仕事でまたジム行けなかった。",
        "誕生日おめでとう！今夜楽しんでね。",
        "隣の人また改装してる、ドリルが止まらない。",
        "おすすめのドラマある？もう見るものない。",
        "今週仕事終わりに泳ぎに行こうよ。",
        "新しいアクアリウム買った、魚はソルトとペッパーって名前。",
        "疲れてるのに眠れない。",
        "職場用のケーキできた、みんなの顔が楽しみ。",
        "こんな灰色の日には温かいお茶に限る。",
    ],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v3() -> list[dict]:
    v2 = [json.loads(line) for line in V2_CORPUS.read_text(encoding="utf-8")
          .splitlines() if line.strip()]
    kept = [r for r in v2 if r.get("category") != "ticket-non-en"]
    dropped = len(v2) - len(kept)
    chat_rows = [{"text": t, "category": "free-chat-multilingual"}
                 for texts in MULTILINGUAL_FREE_CHAT.values() for t in texts]
    rows = kept + chat_rows
    keys = [_norm(r["text"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate texts in corpus v3"
    ban = banned_texts(include_fetch_only=False)
    overlap = [k for k in set(keys) if k in ban]
    assert not overlap, f"contamination: {overlap[:3]}"
    V3_CORPUS.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    print(f"corpus v3: {len(rows)} rows "
          f"(dropped {dropped} ticket-non-en, added {len(chat_rows)} "
          f"multilingual free-chat)")
    return rows


def eval_de_ticket_hashes() -> set[str]:
    """The tickets-de retention slice, recomputed EXACTLY as the eval
    reads it (CSV order, language==de, body>=4 chars, first 300)."""
    seen: set[str] = set()
    taken = 0
    with CSV_PATH.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if taken >= 300:
                break
            if row.get("language") != "de":
                continue
            body = (row.get("body") or "").strip()
            if len(body) >= 4:
                seen.add(_norm(body))
                taken += 1
    assert taken == 300, f"eval slice recomputed {taken} rows, expected 300"
    return seen


def build_tickets(ban: set[str]) -> list[dict]:
    eval_slice = eval_de_ticket_hashes()
    pool: list[str] = []
    pool_keys: set[str] = set()
    with CSV_PATH.open(encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("language") != "de":
                continue
            body = (row.get("body") or "").strip()
            if len(body) < 40:
                continue
            key = _norm(body)
            if key in eval_slice or key in ban or key in pool_keys:
                continue
            pool_keys.add(key)
            pool.append(key)
    rng = random.Random(SEED)
    rng.shuffle(pool)
    took = pool[:N_TICKETS]
    assert len(took) == N_TICKETS, (
        f"only {len(took)} disjoint de tickets available")
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"text": t, "category": "issue"} for t in took]
    TICKETS_OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    print(f"de tickets (in-domain issue rows): {len(rows)} "
          f"(pool {len(pool)}, eval slice excluded)")
    return rows


def main() -> int:
    build_v3()
    ban = banned_texts(include_fetch_only=False)
    build_tickets(ban)
    for path in (V3_CORPUS, TICKETS_OUT):
        path.with_suffix(path.suffix + ".sha256").write_text(
            _sha(path) + "\n", encoding="utf-8")
        print(f"{path.name}: sha256 {_sha(path)[:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
