import bgflow as bg


def parse_architecture_layer_str(
    what_str: str, on_str: str, shape_info: bg.ShapeDictionary
):
    what = []
    on = []

    shape_info_names = [item.name for item in shape_info]
    shape_info = list(shape_info.keys())

    def parse_string(string: str):
        output = []
        string_split = string.split("+")
        for i in range(len(string_split)):
            output.append(shape_info[shape_info_names.index(string_split[i].strip())])
        return output

    if what_str.strip() != "ALL" and what_str.strip() != "REMAINING":
        what = parse_string(what_str)

    if on_str.strip() != "ALL" and on_str.strip() != "REMAINING":
        on = parse_string(on_str)

    if what_str.strip() == "ALL":
        assert (
            on_str.strip() != "REMAINING"
        ), "Cannot have one of 'what' and 'on' as 'ALL' and the other as 'REMAINING'."
        for i in range(len(shape_info)):
            what.append(shape_info[i])

    if on_str.strip() == "ALL":
        assert (
            what_str.strip() != "REMAINING"
        ), "Cannot have one of 'what' and 'on' as 'ALL' and the other as 'REMAINING'."
        for i in range(len(shape_info)):
            on.append(shape_info[i])

    if what_str.strip() == "REMAINING":
        assert (
            on_str.strip() != "REMAINING"
        ), "Cannot have both 'what' and 'on' as 'REMAINING'."
        for i in range(len(shape_info)):
            if shape_info[i] not in on:
                what.append(shape_info[i])

    if on_str.strip() == "REMAINING":
        assert (
            what_str.strip() != "REMAINING"
        ), "Cannot have both 'what' and 'on' as 'REMAINING'."
        for i in range(len(shape_info)):
            if shape_info[i] not in what:
                on.append(shape_info[i])

    return tuple(what), tuple(on)
