from morphokin.data.kinyarwanda import parse_text_to_morpho_sentence, pos_tag_view, norm_text, init_morphokin_socket

if __name__ == '__main__':
    uds_client = init_morphokin_socket(connect_to_flexkin_socket=True, sock_file='/home/ubuntu/MORPHODATA/run/morpho.sock')
    text = "Iyi kipe yatwaye irushanwa rya 2023, yakurikiwe n’iy’Umudage Lukas Baum watwaye Cape Epic mu 2022, kuri ubu uri gukina afatanyije n’Umunya-Kenya Dan Kiptala aho bo bakoresheje iminota 26 n’amasegonda 48."
    normalized_text = norm_text(text)

    print(normalized_text)
