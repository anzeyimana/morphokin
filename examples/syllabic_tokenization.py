from morphokin.data.kinyarwanda import parse_text_to_morpho_sentence, pos_tag_view
from morphokin.data.syllable_vocab import text_to_id_sequence
from morphokin.data.uds_client import UnixSocketClient

if __name__ == '__main__':
    text = "Ikiremwamuntu cyose kivukana umudendezo kandi kingana mu cyubahiro n'uburenganzira. Gifite ubushobozi bwo gutekereza n'umutimanama kandi kigomba gukorera bagenzi bacyo mu mwuka wa kivandimwe."
    id_seq = text_to_id_sequence(text)

    print(id_seq)
