#!/usr/bin/env python3
"""Build the multilingual in-domain ticket file v4 (owner directive:
"everything must be multilingual — that was the whole point").

The corpus-v3 one-shot measured the residual gap: real formal German
tickets route perfectly (retention 1.0, accuracy 0.9933 vs shipped 0.6733)
but the eval's casual-register non-EN support requests still flag
the text class (retention 0.4133, uniform across de/es/fr/ja/pl/pt). The
training data had NO casual-register non-EN work examples — register and
language correlated perfectly with the label. This file fills the missing
quadrant: agent-authored casual work requests (accounts, payments, bugs,
logins, exports, refunds) as ``issue`` rows across the eight served
languages, concatenated with the 1,500 real formal de tickets.

Output: data/real/multilang-tickets/multilang-tickets-v4.jsonl — the
MIVIA_EXTRA_INDOMAIN file for the corpus-v4 run. Disjointness verified by
normalized-text hash against every eval surface and corpus (the v2 OOD
set — whose ticket-non-en rows these must never copy — is in the ban
list); sha-pinned by tests/test_ood_aux_disjoint.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_ood_aux_hf import _norm, banned_texts  # noqa: E402

TICKETS_DIR = ROOT / "data/real/multilang-tickets"
DE_FORMAL = TICKETS_DIR / "multilang-de-tickets.jsonl"
OUT = TICKETS_DIR / "multilang-tickets-v4.jsonl"

# 24 casual-register work requests per language (owner annotation
# directive: the session LLM authors; varied support topics, no
# templates). These are TICKETS (work content), just written the way
# users actually write them.
INFORMAL: dict[str, list[str]] = {
    "de": [
        "Irgendwie ist mein Account gesperrt, keine Ahnung warum. Könnt ihr das aufheben?",
        "Hey, das Passwort-Reset-Mail kommt einfach nicht an. Schon dreimal probiert.",
        "Bei mir stürzt die App ab, sobald ich auf Einstellungen gehe. Pixel 8, aktuell.",
        "Kann ich mein Abo auf den neuen Tarif umstellen oder muss ich kündigen und neu anfangen?",
        "Die Zahlung wurde doppelt abgebucht, kann da mal jemand drüberschauen?",
        "Wie bekomm ich eigentlich die alten Rechnungen als PDF? Finde das nirgends.",
        "Mein Kollege kann das geteilte Projekt nicht öffnen, alle anderen schon.",
        "Seit dem Update lädt der Export ewig, vorher ging das in Sekunden.",
        "Kann man zwei Accounts zusammenlegen? Ich hab versehentlich zwei angelegt.",
        "Login geht bei mir seit heute Morgen nicht mehr, Fehlercode 42.",
        "Die Benachrichtigungen kommen bei mir doppelt, das nervt ziemlich.",
        "Wie lösche ich mein Konto komplett? Bitte alles mit.",
        "Beim Hochladen der Datei kommt immer Serverfehler, Datei ist nur 2 MB.",
        "Kriegt man die App auch auf Deutsch? Bei mir ist alles Englisch.",
        "Meine Karte läuft ab, wo kann ich die neue hinterlegen?",
        "Der Dunkelmodus ist viel zu dunkel, kann man das anpassen?",
        "Suchfunktion findet meine eigenen Beiträge nicht, das ist doch ein Bug oder?",
        "Kann ich den Vertrag schon einen Monat vorher kündigen? Zieh um.",
        "Beim Bezahlen mit PayPal bricht er immer ab, mit Karte geht's.",
        "Wo sehe ich, wer alles im Team ist? Finde die Mitgliederliste nicht.",
        "Die Verifizierungs-SMS kommt stundenlang nicht, sitze hier fest.",
        "Kann man den Newsletter abstellen ohne das Konto zu löschen?",
        "Meine Daten sind weg nach dem Update, vorher hatte ich hunderte Einträge.",
        "Gibt es einen Export ins CSV-Format? Brauche das für die Steuer.",
    ],
    "fr": [
        "Mon compte est bloqué sans raison, vous pouvez débloquer ?",
        "Le mail de réinitialisation n'arrive pas, j'ai essayé trois fois.",
        "L'appli plante dès que j'ouvre les réglages, sur un Pixel 8 à jour.",
        "Je peux passer mon abo au nouveau tarif ou il faut résilier et recommencer ?",
        "Le prélèvement est passé en double, quelqu'un peut vérifier ?",
        "Comment récupérer les anciennes factures en PDF ? Je trouve nulle part.",
        "Mon collègue n'arrive pas à ouvrir le projet partagé, tous les autres si.",
        "Depuis la mise à jour l'export met une éternité, avant c'était instantané.",
        "On peut fusionner deux comptes ? J'en ai créé deux par erreur.",
        "Impossible de me connecter depuis ce matin, code erreur 42.",
        "Je reçois les notifications en double, c'est assez agaçant.",
        "Comment supprimer définitivement mon compte ? Avec toutes les données.",
        "À chaque upload j'ai erreur serveur, le fichier fait 2 Mo.",
        "On peut avoir l'appli en français ? Chez moi tout est en anglais.",
        "Ma carte expire, où est-ce que je peux mettre la nouvelle ?",
        "Le mode sombre est trop sombre, on peut régler ça ?",
        "La recherche ne retrouve pas mes propres posts, c'est un bug non ?",
        "Je peux résilier un mois avant la fin ? Je déménage.",
        "Le paiement PayPal échoue tout le temps, par carte ça marche.",
        "Où voir qui est dans l'équipe ? Je ne trouve plus la liste.",
        "Le SMS de vérification n'arrive pas depuis des heures, je suis bloqué.",
        "On peut désactiver la newsletter sans supprimer le compte ?",
        "Mes données ont disparu après la mise à jour, j'avais des centaines d'entrées.",
        "Il existe un export CSV ? J'en ai besoin pour les impôts.",
    ],
    "es": [
        "Mi cuenta está bloqueada y no sé por qué. ¿Pueden desbloquearla?",
        "El correo para restablecer la contraseña no llega, ya lo intenté tres veces.",
        "La app se cierra al abrir ajustes, tengo un Pixel 8 actualizado.",
        "¿Puedo cambiar mi suscripción al nuevo plan o tengo que cancelar y empezar de cero?",
        "Me cobraron dos veces el mismo pago, ¿alguien puede revisarlo?",
        "¿Cómo descargo las facturas antiguas en PDF? No lo encuentro.",
        "Mi compañero no puede abrir el proyecto compartido, todos los demás sí.",
        "Desde la actualización la exportación tarda muchísimo, antes era al instante.",
        "¿Se pueden fusionar dos cuentas? Creé dos sin querer.",
        "No puedo iniciar sesión desde esta mañana, código de error 42.",
        "Me llegan las notificaciones duplicadas, es bastante molesto.",
        "¿Cómo elimino mi cuenta por completo? Con todos los datos.",
        "Al subir el archivo siempre da error de servidor, pesa solo 2 MB.",
        "¿Se puede tener la app en español? A mí me sale todo en inglés.",
        "Mi tarjeta caduca, ¿dónde puedo poner la nueva?",
        "El modo oscuro es demasiado oscuro, ¿se puede ajustar?",
        "El buscador no encuentra mis propias publicaciones, ¿es un bug no?",
        "¿Puedo cancelar el contrato un mes antes? Me mudo.",
        "El pago con PayPal siempre falla, con tarjeta funciona.",
        "¿Dónde veo quién está en el equipo? No encuentro la lista.",
        "El SMS de verificación tarda horas, estoy bloqueado.",
        "¿Se puede quitar el boletín sin borrar la cuenta?",
        "Perdí mis datos tras la actualización, tenía cientos de registros.",
        "¿Hay exportación a CSV? Lo necesito para la declaración.",
    ],
    "pl": [
        "Konto mi zablokowane i nie wiem dlaczego. Możecie odblokować?",
        "Mail z resetem hasła nie przychodzi, próbowałem już trzy razy.",
        "Aplikacja wyłącza się przy wejściu w ustawienia, Pixel 8, wszystko aktualne.",
        "Mogę przejść na nowy abonament czy muszę anulować i zacząć od nowa?",
        "Pobrano mi płatność podwójnie, ktoś to sprawdzi?",
        "Jak pobrać stare faktury w PDF? Nigdzie nie mogę znaleźć.",
        "Kolega nie może otworzyć udostępnionego projektu, reszta może.",
        "Od aktualizacji eksport trwa wieczność, wcześniej było od razu.",
        "Można połączyć dwa konta? Przez przypadek założyłem dwa.",
        "Nie mogę się zalogować od rana, kod błędu 42.",
        "Powiadomienia przychodzą podwójnie, całkiem to irytuje.",
        "Jak usunąć konto całkowicie? Wraz ze wszystkimi danymi.",
        "Przy wgrywaniu pliku zawsze błąd serwera, plik ma tylko 2 MB.",
        "Da się mieć aplikację po polsku? U mnie wszystko po angielsku.",
        "Moja karta wygasa, gdzie mogę wpisać nową?",
        "Tryb ciemny jest za ciemny, da się to regulować?",
        "Wyszukiwarka nie znajduje moich własnych postów, to bug nie?",
        "Mogę rozwiązać umowę miesiąc wcześniej? Przeprowadzam się.",
        "Płatność PayPalem zawsze się nie udaje, kartą działa.",
        "Gdzie zobaczę, kto jest w zespole? Nie mogę znaleźć listy.",
        "SMS z weryfikacją nie przychodzi od godzin, utknąłem.",
        "Da się wyłączyć newsletter bez usuwania konta?",
        "Zniknęły mi dane po aktualizacji, miałem setki wpisów.",
        "Jest eksport do CSV? Potrzebne do rozliczenia podatkowego.",
    ],
    "pt": [
        "A minha conta está bloqueada e não sei porquê. Podem desbloquear?",
        "O email para repor a password não chega, já tentei três vezes.",
        "A app fecha quando abro as definições, tenho um Pixel 8 atualizado.",
        "Posso mudar a subscrição para o novo plano ou tenho de cancelar e recomeçar?",
        "Fui cobrado duas vezes pelo mesmo pagamento, alguém pode ver isso?",
        "Como saco as faturas antigas em PDF? Não encontro.",
        "O meu colega não consegue abrir o projeto partilhado, os outros sim.",
        "Desde a atualização a exportação demora uma eternidade.",
        "Dá para juntar duas contas? Criei duas sem querer.",
        "Não consigo entrar desde esta manhã, código de erro 42.",
        "Recebo as notificações duplicadas, é bastante chato.",
        "Como elimino a conta por completo? Com todos os dados.",
        "Ao carregar o ficheiro dá sempre erro de servidor, só tem 2 MB.",
        "Dá para ter a app em português? A mim aparece tudo em inglês.",
        "O meu cartão expira, onde posso pôr o novo?",
        "O modo escuro é escuro demais, dá para ajustar?",
        "A pesquisa não encontra as minhas próprias publicações, é um bug não?",
        "Posso cancelar o contrato um mês antes? Mudo de casa.",
        "O pagamento por PayPal falha sempre, com cartão funciona.",
        "Onde vejo quem está na equipa? Não encontro a lista.",
        "O SMS de verificação demora horas, estou preso.",
        "Dá para desligar a newsletter sem apagar a conta?",
        "Perdi os dados depois da atualização, tinha centenas de registos.",
        "Existe exportação para CSV? Preciso para o imposto.",
    ],
    "it": [
        "Il mio account è bloccato e non capisco perché. Potete sbloccarlo?",
        "La mail per reimpostare la password non arriva, ho già provato tre volte.",
        "L'app si chiude appena apro le impostazioni, Pixel 8 aggiornato.",
        "Posso passare al nuovo abbonamento o devo disdire e ricominciare?",
        "Mi hanno addebitato il pagamento due volte, qualcuno può controllare?",
        "Come scarico le vecchie fatture in PDF? Non le trovo da nessuna parte.",
        "Il mio collega non riesce ad aprire il progetto condiviso, tutti gli altri sì.",
        "Dall'aggiornamento l'esportazione mette un'eternità, prima era immediata.",
        "Si possono unire due account? Ne ho creati due per sbaglio.",
        "Non riesco ad accedere da stamattina, codice errore 42.",
        "Le notifiche mi arrivano doppie, è parecchio fastidioso.",
        "Come elimino definitivamente l'account? Con tutti i dati.",
        "Caricando il file dà sempre errore del server, pesa solo 2 MB.",
        "Si può avere l'app in italiano? Da me è tutto in inglese.",
        "La mia carta scade, dove posso inserire la nuova?",
        "La modalità scura è troppo scura, si può regolare?",
        "La ricerca non trova i miei post, è un bug no?",
        "Posso disdire il contratto un mese prima? Mi trasferisco.",
        "Il pagamento con PayPal fallisce sempre, con la carta funziona.",
        "Dove vedo chi è nel team? Non trovo più l'elenco.",
        "L'SMS di verifica non arriva da ore, sono bloccato.",
        "Si può disattivare la newsletter senza cancellare l'account?",
        "Ho perso i dati dopo l'aggiornamento, avevo centinaia di voci.",
        "C'è un'esportazione CSV? Mi serve per la dichiarazione.",
    ],
    "nl": [
        "Mijn account is geblokkeerd en ik snap niet waarom. Kunnen jullie het deblokkeren?",
        "De wachtwoord-resetmail komt gewoon niet aan, al drie keer geprobeerd.",
        "De app crasht zodra ik instellingen open, Pixel 8, helemaal bijgewerkt.",
        "Kan ik mijn abonnement overstappen naar het nieuwe pakket of moet ik opzeggen?",
        "De betaling is twee keer afgeschreven, kan iemand ernaar kijken?",
        "Hoe download ik oude facturen als PDF? Ik kan het nergens vinden.",
        "Mijn collega kan het gedeelde project niet openen, alle anderen wel.",
        "Sinds de update doet de export er eeuwen over, eerst was het direct.",
        "Kunnen twee accounts samengevoegd worden? Ik heb er per ongeluk twee.",
        "Inloggen lukt sinds vanochtend niet meer, foutcode 42.",
        "Ik krijg meldingen dubbel, dat is best irritant.",
        "Hoe verwijder ik mijn account volledig? Met alle data erbij.",
        "Bij het uploaden komt steeds een serverfout, het bestand is maar 2 MB.",
        "Kan de app ook in het Nederlands? Bij mij is alles Engels.",
        "Mijn kaart verloopt, waar kan ik de nieuwe invoeren?",
        "De donkere modus is veel te donker, kan dat aangepast worden?",
        "De zoekfunctie vindt mijn eigen berichten niet, is dat geen bug?",
        "Kan ik het contract een maand eerder opzeggen? Ik verhuis.",
        "Betalen met PayPal lukt nooit, met kaart wel.",
        "Waar zie ik wie er in het team zitten? Ik mis de ledenlijst.",
        "De verificatie-sms komt al uren niet opdagen, ik zit vast.",
        "Kan de nieuwsbrief uit zonder account te verwijderen?",
        "Mijn data is weg na de update, ik had honderden invoer.",
        "Is er een CSV-export? Ik heb het nodig voor de belasting.",
    ],
    "ja": [
        "アカウントがロックされて理由がわかりません。解除できますか？",
        "パスワード再設定のメールが届きません。三回試しました。",
        "設定を開くとアプリが落ちます。Pixel 8で最新版です。",
        "新しいプランに変更できますか？それとも解約してやり直しですか？",
        "支払いが二回引き落とされています。確認してもらえますか？",
        "前の請求書をPDFで入手するには？どこにも見つかりません。",
        "同僚が共有プロジェクトを開けません。他の人は開けます。",
        "アップデート後、エクスポートに時間がかかります。前は一瞬だったのに。",
        "二つのアカウントを統合できますか？間違えて二つ作りました。",
        "今朝からログインできません。エラーコード42です。",
        "通知が二重に届きます。ちょっと困ります。",
        "アカウントを完全に削除するには？データも全部消してください。",
        "ファイルをアップロードすると毎回サーバーエラーです。2MBしかないのに。",
        "アプリを日本語にできますか？全部英語なんですが。",
        "カードの有効期限が切れます。新しいカードはどこで登録できますか？",
        "ダークモードが暗すぎます。調整できますか？",
        "検索で自分の投稿が見つかりません。バグじゃないですか？",
        "契約を一ヶ月早く解約できますか？引っ越しするので。",
        "PayPalでの支払いがいつも失敗します。カードなら大丈夫です。",
        "チームのメンバーはどこで見れますか？一覧が見つかりません。",
        "認証SMSが何時間も届きません。先に進めません。",
        "ニュースレターだけ停止できますか？アカウントは残したいです。",
        "アップデート後、データが消えました。何百も登録していたのに。",
        "CSVエクスポートはありますか？確定申告に必要です。",
    ],
}


def main() -> int:
    ban = banned_texts(include_fetch_only=False)
    informal_rows = []
    for lang, texts in INFORMAL.items():
        assert len(texts) >= 24, f"{lang}: {len(texts)} rows, need >= 24"
        for t in texts:
            key = _norm(t)
            assert key not in ban, f"{lang} row collides with eval text: {t[:50]}"
            informal_rows.append({"text": t, "category": "issue"})
    keys = [_norm(r["text"]) for r in informal_rows]
    assert len(keys) == len(set(keys)), "duplicate informal rows"

    formal = [json.loads(line) for line in DE_FORMAL.read_text(encoding="utf-8")
              .splitlines() if line.strip()]
    rows = formal + informal_rows
    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix(OUT.suffix + ".sha256").write_text(digest + "\n",
                                                       encoding="utf-8")
    print(f"multilang-tickets-v4.jsonl: {len(rows)} rows "
          f"({len(formal)} formal de + {len(informal_rows)} informal "
          f"across {len(INFORMAL)} languages), sha256 {digest[:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
