#!/usr/bin/env python3
"""v5 multilingual ticket extension: consumer-commerce support requests.

The corpus-v4 one-shot isolated the last flagged quadrant: formal-polite
non-EN support requests about orders, deliveries, refunds, damaged goods
and card declines (29/150 rows; retention 0.8067 vs the 0.99 bar) —
content with zero in-domain examples in any language. These rows fill it
as ``issue`` examples, REQUEST/COMPLAINT style (asks for action), never
order-confirmation records: the v2 eval's receipt/email categories stay
separable ("Order confirmation: ... arriving Tuesday" is a record, not a
request).

Output: data/real/multilang-tickets/multilang-tickets-v5.jsonl = v4 +
the new rows. Disjointness verified against every eval surface and
corpus; sha-pinned by tests/test_ood_aux_disjoint.py.
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
V4 = TICKETS_DIR / "multilang-tickets-v4.jsonl"
OUT = TICKETS_DIR / "multilang-tickets-v5.jsonl"

COMMERCE: dict[str, list[str]] = {
    "de": [
        "Guten Tag, die App stürzt seit gestern Abend immer ab, sobald ich mein Profil öffne.",
        "Ich habe schon drei Nachrichten geschrieben, aber nie eine Antwort bekommen. Warum?",
        "Beim Hochladen meiner Fotos bricht es immer bei siebzig Prozent ab. Was kann ich tun?",
        "Meine Bestellung sollte Montag kommen, der Status steht seit Tagen auf unterwegs.",
        "Ich wurde diesen Monat doppelt abgerechnet. Wann bekomme ich die Erstattung?",
        "Der Karton war beschädigt und ein Teil fehlt. Wie läuft die Ersatzlieferung?",
        "Meine Karte wird seit Freitag abgelehnt, in anderen Shops geht sie problemlos.",
        "Können Sie mich bitte zurückrufen? Der Chat bricht bei mir immer ab.",
        "Ich habe das falsche Paket bekommen, Aufkleber mit Namen vom Nachbarn.",
        "Wie kann ich den Rücksendeschein ausdrucken? Der Link in der Mail ist tot.",
        "Die Sitzung wird bei jedem Klick beendet und ich muss neu einloggen.",
        "Guten Tag, die Erstattung für die Rückgabe ist seit zwei Wochen offen.",
        "Mein Gutscheincode wird nicht angenommen, obwohl er noch gültig ist.",
        "Können Sie die Lieferung an eine Packstation umleiten? Ich bin nächste Woche weg.",
        "Die Rechnungsadresse auf der Bestellung ist falsch, bitte korrigieren.",
        "Zwei Positionen meiner Bestellung fehlen, der Rest ist da.",
        "Wann kommt denn der Rest meiner Bestellung? Es steht nur teilweise geliefert.",
        "Die Sendungsverfolgung zeigt seit drei Tagen nichts Neues mehr.",
        "Ich möchte den defekten Artikel umtauschen, wie gehe ich vor?",
        "Guten Morgen, mein Kundenkonto zeigt ein Fremd-Lager, ich bin es nicht.",
        "Die App sagt mir ständig, mein Warenkorb sei leer, obwohl Artikel drin sind.",
        "Kann ich die Bestellung noch ändern? Sie ist noch nicht versandt.",
        "Die Rückholung des Pakets wurde zweimal nicht abgeholt.",
        "Meine Nachricht an den Verkäufer bleibt unbeantwortet, Frist läuft bald ab.",
        "Der Gutschein wurde abgebucht, die Bestellung aber storniert.",
        "Ich habe versehentlich die Gäste-Adresse als Lieferadresse gewählt.",
        "Bitte um Stornierung meiner gestrigen Bestellung, sie ist noch offen.",
        "Der Kundenservice-Chat wirft mich nach zwei Minuten immer raus.",
        "Die Erstattung ist nur teilweise erfolgt, der Versand wurde nicht erstattet.",
        "Wo bleibt meine Bestellung aus Kalenderwoche 40? Keine Statusänderung.",
    ],
    "fr": [
        "Bonjour, l'appli plante depuis hier soir dès que j'ouvre mon profil.",
        "J'ai déjà écrit trois messages, jamais de réponse. Pourquoi ?",
        "L'envoi de mes photos plante toujours à soixante-dix pour cent.",
        "Ma commande devait arriver lundi, le statut est bloqué sur en transit.",
        "J'ai été débité deux fois ce mois-ci. Quand aurai-je le remboursement ?",
        "Le carton était abîmé et une pièce manque. Comment se passe le remplacement ?",
        "Ma carte est refusée depuis vendredi, elle marche partout ailleurs.",
        "Pouvez-vous me rappeler ? Le chat se coupe sans arrêt.",
        "J'ai reçu le colis d'un voisin, l'étiquette n'était pas pour moi.",
        "Comment imprimer l'étiquette de retour ? Le lien du mail est mort.",
        "La session se ferme à chaque clic et je dois me reconnecter.",
        "Bonjour, le remboursement du retour est en attente depuis deux semaines.",
        "Mon code promo est refusé alors qu'il est encore valide.",
        "Pouvez-vous rediriger la livraison vers un point relais ? Je pars.",
        "L'adresse de facturation sur la commande est fausse, merci de corriger.",
        "Deux articles de ma commande manquent, le reste est bien là.",
        "Quand arrive le reste de ma commande ? Statut : partiellement livré.",
        "Le suivi colis n'affiche rien de neuf depuis trois jours.",
        "Je souhaite échanger l'article défectueux, quelle est la démarche ?",
        "Bonjour, mon compte client affiche un entrepôt étranger, ce n'est pas moi.",
        "L'appli me dit que mon panier est vide alors qu'il y a des articles.",
        "Puis-je encore modifier la commande ? Elle n'est pas expédiée.",
        "L'enlèvement du colis a échoué deux fois de suite.",
        "Mon message au vendeur reste sans réponse, le délai expire bientôt.",
        "Le bon d'achat a été débité mais la commande annulée.",
        "J'ai mis par erreur l'adresse de mes invités en adresse de livraison.",
        "Merci d'annuler ma commande d'hier, elle est encore ouverte.",
        "Le chat du service client me rejette au bout de deux minutes.",
        "Le remboursement n'est que partiel, les frais de port n'ont pas été rendus.",
        "Où en est ma commande de la semaine 40 ? Aucun changement de statut.",
    ],
    "es": [
        "Buenas tardes, la aplicación se cierra desde anoche al abrir mi perfil.",
        "Ya escribí tres mensajes y nunca recibí respuesta. ¿Por qué?",
        "Al subir mis fotos siempre se corta al setenta por ciento. ¿Qué hago?",
        "Mi pedido llegaba el lunes y el estado lleva días en tránsito.",
        "Me cobraron dos veces este mes. ¿Cuándo me devuelven el dinero?",
        "La caja llegó dañada y falta una pieza. ¿Cómo funciona el reemplazo?",
        "Mi tarjeta es rechazada desde el viernes, en otras tiendas funciona.",
        "¿Pueden llamarme? El chat se me corta continuamente.",
        "Recibí el paquete de mi vecino, la etiqueta no era la mía.",
        "¿Cómo imprimo la etiqueta de devolución? El enlace no funciona.",
        "La sesión se cierra con cada clic y tengo que iniciar de nuevo.",
        "Buenas tardes, el reembolso de la devolución lleva dos semanas pendiente.",
        "Mi código promocional no se acepta aunque sigue vigente.",
        "¿Pueden redirigir el envío a un punto de recogida? Me voy de viaje.",
        "La dirección de facturación del pedido es incorrecta, por favor corríjanla.",
        "Faltan dos artículos de mi pedido, el resto sí llegó.",
        "¿Cuándo llega el resto del pedido? Aparece como entregado parcialmente.",
        "El seguimiento del paquete no muestra nada nuevo desde hace tres días.",
        "Quiero cambiar el artículo defectuoso, ¿cómo lo hago?",
        "Buenos días, mi cuenta muestra un almacén extranjero que no es mío.",
        "La app dice que mi carrito está vacío cuando tiene artículos.",
        "¿Puedo modificar el pedido todavía? No se ha enviado.",
        "La recogida del paquete falló dos veces seguidas.",
        "Mi mensaje al vendedor sigue sin respuesta, el plazo vence pronto.",
        "Cobraron el vale pero el pedido quedó cancelado.",
        "Puse por error la dirección de mis invitados como dirección de entrega.",
        "Ruego cancelen mi pedido de ayer, sigue abierto.",
        "El chat de atención al cliente me expulsa a los dos minutos.",
        "El reembolso fue parcial, no devolvieron los gastos de envío.",
        "¿Dónde está mi pedido de la semana 40? Sin cambios de estado.",
    ],
    "pl": [
        "Dzień dobry, aplikacja wyłącza się od wczorajszego wieczoru przy otwarciu profilu.",
        "Pisałem już trzy wiadomości i żadnej odpowiedzi. Dlaczego?",
        "Przy wgrywaniu zdjęć zawsze urywa się na siedemdziesięciu procentach.",
        "Paczka miała być w poniedziałek, a status od dni stoi w drodze.",
        "Dwukrotnie pobrano opłatę w tym miesiącu. Kiedy zwrot?",
        "Karton był uszkodzony i brakuje części. Jak wygląda wymiana?",
        "Moja karta jest odrzucana od piątku, w innych sklepach działa.",
        "Możecie oddzwonić? Czat ciągle mi się ucina.",
        "Przyszła mi paczka sąsiadki, a jej moja. Jak je zamienić?",
        "Jak wydrukować etykietę zwrotu? Link z maila nie działa.",
        "Sesja kończy się przy każdym kliknięciu i muszę logować się od nowa.",
        "Dzień dobry, zwrot za przesyłkę wisi od dwóch tygodni.",
        "Mój kod rabatowy nie działa, choć jest jeszcze ważny.",
        "Możecie przekierować paczkę do paczkomatu? Wyjeżdżam.",
        "Adres na fakturze jest błędny, proszę poprawić.",
        "Brakuje dwóch pozycji z zamówienia, reszta doszła.",
        "Kiedy będzie reszta zamówienia? Status: dostarczono częściowo.",
        "Śledzenie paczki nie pokazuje nic nowego od trzech dni.",
        "Chcę wymienić uszkodzony artykuł, jak to zrobić?",
        "Dzień dobry, na koncie widzę obcy magazyn, to nie mój.",
        "Aplikacja mówi, że koszyk jest pusty, a są w nim artykuły.",
        "Mogę jeszcze zmienić zamówienie? Nie zostało wysłane.",
        "Odbiór paczki nie odbył się dwa razy z rzędu.",
        "Wiadomość do sprzedawcy bez odpowiedzi, termin się kończy.",
        "Voucher pobrano, ale zamówienie anulowano.",
        "Przez pomyłkę podałem adres gości jako adres dostawy.",
        "Proszę o anulowanie wczorajszego zamówienia, wciąż otwarte.",
        "Czat obsługi wyrzuca mnie po dwóch minutach.",
        "Zwrot tylko częściowy, kosztów wysyłki nie zwrócono.",
        "Gdzie jest moje zamówienie z czterdziestego tygodnia? Bez zmian.",
    ],
    "pt": [
        "Boa tarde, a app fecha desde ontem à noite quando abro o perfil.",
        "Já escrevi três mensagens e nunca obtive resposta. Porquê?",
        "Ao carregar as fotos rebenta sempre aos setenta por cento. O que faço?",
        "A minha encomenda chegava segunda-feira e o estado está parado há dias.",
        "Fui debitado duas vezes este mês. Quando recebo o reembolso?",
        "A caixa veio danificada e falta uma peça. Como funciona a substituição?",
        "O meu cartão é recusado desde sexta, nas outras lojas funciona.",
        "Podem ligar-me? O chat corta-me sempre.",
        "Recebi a encomenda do vizinho, a etiqueta não era a minha.",
        "Como imprimo a etiqueta de devolução? A ligação do email não abre.",
        "A sessão fecha a cada clique e tenho de iniciar sessão outra vez.",
        "Boa tarde, o reembolso da devolução está pendente há duas semanas.",
        "O meu código promocional não é aceite embora ainda seja válido.",
        "Podem redirecionar a entrega para um ponto de recolha? Viajo.",
        "O endereço de faturação da encomenda está errado, corrijam por favor.",
        "Faltam dois artigos da minha encomenda, o resto chegou.",
        "Quando chega o resto da encomenda? Aparece entregue parcialmente.",
        "O rastreio da encomenda não mostra novidades há três dias.",
        "Quero trocar o artigo defeituoso, como procedo?",
        "Bom dia, a minha conta mostra um armazém estrangeiro que não é meu.",
        "A app diz que o carrinho está vazio quando tem artigos.",
        "Posso ainda alterar a encomenda? Ainda não foi enviada.",
        "A recolha da encomenda falhou duas vezes seguidas.",
        "A minha mensagem ao vendedor continua sem resposta, o prazo expira.",
        "O vale foi debitado mas a encomenda cancelada.",
        "Pus por engano a morada dos meus convidados como endereço de entrega.",
        "Peço o cancelamento da encomenda de ontem, ainda está aberta.",
        "O chat do apoio ao cliente expulsa-me ao fim de dois minutos.",
        "O reembolso foi parcial, não devolveram os portes.",
        "Onde está a minha encomenda da semana quarenta? Sem alterações.",
    ],
    "it": [
        "Buongiorno, l'app si chiude da ieri sera appena apro il profilo.",
        "Ho già scritto tre messaggi e nessuna risposta. Perché?",
        "Caricando le foto si interrompe sempre al settanta per cento.",
        "Il mio ordine doveva arrivare lunedì, lo stato è fermo da giorni.",
        "Mi hanno addebitato due volte questo mese. Quando arriva il rimborso?",
        "La scatola era danneggiata e manca un pezzo. Come funziona la sostituzione?",
        "La mia carta viene rifiutata da venerdì, negli altri negozi funziona.",
        "Potete richiamarmi? La chat mi si chiude in continuazione.",
        "Ho ricevuto il pacco di un vicino, l'etichetta non era la mia.",
        "Come stampo l'etichetta di reso? Il link nella mail non funziona.",
        "La sessione si chiude a ogni clic e devo rifare il login.",
        "Buongiorno, il rimborso del reso è in sospeso da due settimane.",
        "Il mio codice promo non viene accettato anche se è ancora valido.",
        "Potete dirottare la consegna su un punto di ritiro? Parto.",
        "L'indirizzo di fatturazione sull'ordine è sbagliato, correggetelo.",
        "Mancano due articoli del mio ordine, il resto è arrivato.",
        "Quando arriva il resto dell'ordine? Risulta consegnato parzialmente.",
        "Il tracking del pacco non mostra novità da tre giorni.",
        "Voglio cambiare l'articolo difettoso, come faccio?",
        "Buongiorno, sul mio account compare un magazzino estero, non è mio.",
        "L'app dice che il carrello è vuoto quando invece ha articoli.",
        "Posso ancora modificare l'ordine? Non è stato spedito.",
        "Il ritiro del pacco è fallito due volte di seguito.",
        "Il mio messaggio al venditore resta senza risposta, la scadenza si avvicina.",
        "Il buono è stato addebitato ma l'ordine annullato.",
        "Ho messo per sbaglio l'indirizzo dei miei ospiti come indirizzo di consegna.",
        "Chiedo la disdetta dell'ordine di ieri, è ancora aperto.",
        "La chat dell'assistenza mi espelle dopo due minuti.",
        "Il rimborso è solo parziale, le spese di spedizione non sono state rese.",
        "Dov'è il mio ordine della settimana quaranta? Nessun aggiornamento.",
    ],
    "nl": [
        "Goedemiddag, de app crasht sinds gisteravond zodra ik mijn profiel open.",
        "Ik schreef al drie berichten en nooit antwoord. Waarom niet?",
        "Bij het uploaden van foto's breekt hij altijd op zeventig procent af.",
        "Mijn bestelling zou maandag komen, de status staat al dagen op onderweg.",
        "Dit maandag dubbel afgeschreven. Wanneer krijg ik de terugbetaling?",
        "De doos was beschadigd en er ontbreekt een onderdeel. Hoe zit dat met vervanging?",
        "Mijn kaart wordt sinds vrijdag geweigerd, in andere winkels gaat het prima.",
        "Kunnen jullie mij bellen? De chat valt telkens weg.",
        "Ik kreeg het pakket van de buren, het label was niet voor mij.",
        "Hoe print ik het retourlabel? De link in de mail doet niets.",
        "De sessie sluit bij elke klik en ik moet opnieuw inloggen.",
        "Goedemiddag, de terugbetaling van de retour staat al twee weken open.",
        "Mijn kortingscode wordt niet geaccepteerd terwijl hij nog geldig is.",
        "Kunnen jullie de bezorging naar een ophaalpunt sturen? Ik ga weg.",
        "Het factuuradres op de bestelling klopt niet, graag corrigeren.",
        "Er ontbreken twee artikelen uit mijn bestelling, de rest is er wel.",
        "Wanneer komt de rest van mijn bestelling? Status: deels geleverd.",
        "De track-and-trace laat al drie dagen niets nieuws zien.",
        "Ik wil het kapotte artikel omruilen, hoe werkt dat?",
        "Goedemorgen, op mijn account staat een vreemde magazijn, niet die van mij.",
        "De app zegt dat mijn mandje leeg is terwijl er artikelen in staan.",
        "Kan ik de bestelling nog aanpassen? Hij is nog niet verzonden.",
        "Het ophalen van het pakket is twee keer niet gekomen.",
        "Mijn bericht aan de verkoper blijft onbeantwoord, de termijn loopt af.",
        "De cadeaubon is afgeschreven maar de bestelling geannuleerd.",
        "Ik heb per ongeluk het adres van mijn gasten als bezorgadres ingevuld.",
        "Graag annulering van mijn bestelling van gisteren, hij staat nog open.",
        "De klantenservice-chat gooit me er na twee minuten weer uit.",
        "De terugbetaling was gedeeltelijk, verzendkosten zijn niet terugbetaald.",
        "Waar blijft mijn bestelling van week veertig? Geen statuswijziging.",
    ],
    "ja": [
        "お世話になっております。昨晩からプロフィールを開くとアプリが落ちます。",
        "すでに三回問い合わせましたが、一度も返事がありません。なぜでしょうか。",
        "写真のアップロードが常に七十パーセントで失敗します。どうすればいいですか。",
        "注文は月曜着のはずが、状況が数日間「配送中」のままです。",
        "今月は二回引き落とされています。いつ返金されますか？",
        "箱が破損していて一部の部品がありません。交換はどう進めればいいですか。",
        "金曜からカードが拒否されます。他のショップでは普通に使えます。",
        "お電話いただけますか？チャットが毎回切れてしまいます。",
        "近所の人宛の荷物が届きました。ラベルが私のではありませんでした。",
        "返品ラベルはどう印刷すればいいですか？メールのリンクが開きません。",
        "クリックするたびにセッションが切れて、毎回ログインし直します。",
        "お世話になっております。返品の返金が二週間保留になったままです。",
        "まだ有効なはずのクーポンコードが使えません。",
        "来週不在なので、受け取り場所に配送を変更できますか？",
        "注文の請求先住所が間違っています。修正をお願いします。",
        "注文の二点が届いていません。残りは届いています。",
        "残りの注文はいつ届きますか？「一部配送済み」のままです。",
        "追跡情報が三日間更新されていません。",
        "不具合のある商品を交換したいのですが、手順を教えてください。",
        "おはようございます。アカウントに見覚えのない海外倉庫が表示されます。",
        "カートに商品が入っているのに空だと表示されます。",
        "まだ発送されていないので、注文を変更できますか？",
        "集荷が二回連続で来ませんでした。",
        "出品者へのメッセージの返事がなく、期限が近づいています。",
        "ギフトカードは引き落とされたのに注文はキャンセルされました。",
        "誤って友人の住所を配送先に設定してしまいました。",
        "昨日の注文をキャンセルしてください。まだ未発送です。",
        "カスタマーサポートのチャットが二分で切れて追い出されます。",
        "返金が一部だけで、送料は返ってきていません。",
        "四十週目の注文はどこですか？状況が変わりません。",
    ],
}


def main() -> int:
    ban = banned_texts(include_fetch_only=False)
    new_rows = []
    for lang, texts in COMMERCE.items():
        assert len(texts) >= 30, f"{lang}: {len(texts)} rows, need >= 30"
        for t in texts:
            key = _norm(t)
            assert key not in ban, f"{lang} row collides with eval text: {t[:50]}"
            new_rows.append({"text": t, "category": "issue"})
    keys = [_norm(r["text"]) for r in new_rows]
    assert len(keys) == len(set(keys)), "duplicate commerce rows"

    prev = [json.loads(line) for line in V4.read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    rows = prev + new_rows
    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix(OUT.suffix + ".sha256").write_text(digest + "\n",
                                                       encoding="utf-8")
    print(f"multilang-tickets-v5.jsonl: {len(rows)} rows "
          f"(v4 {len(prev)} + {len(new_rows)} commerce), "
          f"sha256 {digest[:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
