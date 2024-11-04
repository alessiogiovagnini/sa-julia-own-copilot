import tree_sitter_languages as tsl

language = tsl.get_language(language="julia")
parser = tsl.get_parser(language="julia")


if __name__ == '__main__':
    julia_function: str = """
    function Base.getproperty(::SupervisedScitype{input_scitype, target_scitype, prediction_type},
                              field::Symbol) where {input_scitype, target_scitype, prediction_type}
        if field === :input_scitype
            return input_scitype
        elseif field === :target_scitype
            return target_scitype
        elseif field === :prediction_type
            return prediction_type
        else
            throw(ArgumentError("Unsupported property. "))
        end
    end
    """

    tree = parser.parse(julia_function.encode())
    node = tree.root_node
    pass



