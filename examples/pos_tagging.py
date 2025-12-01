from morphokin.data.kinyarwanda import parse_text_to_morpho_sentence, init_morphokin_socket, ParsedFlexSentence, pos_tag_view
from morphokin.data.uds_client import UnixSocketClient

if __name__ == '__main__':
    uds_client: UnixSocketClient = init_morphokin_socket(connect_to_flexkin_socket=True, sock_file='/home/ubuntu/MORPHODATA/run/morpho.sock')
    text: str = "Ikiremwamuntu cyose kivukana umudendezo kandi kingana mu cyubahiro n'uburenganzira. Gifite ubushobozi bwo gutekereza n'umutimanama kandi kigomba gukorera bagenzi bacyo mu mwuka wa kivandimwe."
    parsed: ParsedFlexSentence = parse_text_to_morpho_sentence(uds_client, text)

    print('\n'.join([pos_tag_view(t.pos_tag_id) for t in parsed.tokens]))
