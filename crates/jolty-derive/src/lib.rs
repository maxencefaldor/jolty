//! The procedural macros of jolty.
//!
//! [`Lift`](macro@Lift) writes `jolty_models::Lift` for a type generic over its kind of
//! number, `R`: the type of `R`s made from the same type of plain numbers, `f64`s, each
//! number by the path serde writes it at, so that a twin seeds a parameter's derivatives
//! where the parameter is named, with no detour through the parameters' written form.
//!
//! [`binned_parameters`](macro@binned_parameters) declares a geometry-binned parameter struct
//! (such as BSIM3's and BSIM4's), its `Default` implementation, its companion binned struct,
//! and the `binned` method that adds each binned parameter's length, width, and area
//! dependences.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Tokens, TokenTree};
use quote::quote;
use syn::meta::ParseNestedMeta;
use syn::parse::{Parse, ParseStream};
use syn::{
    Attribute, Data, DeriveInput, Error, Fields, GenericArgument, GenericParam, Generics, Ident,
    LitStr, Path, PathArguments, Result, ReturnType, Token, Type, TypeParamBound, Visibility,
    braced, parenthesized, parse_macro_input,
};

/// Writes `jolty_models::Lift<R>` for a struct or an enum generic over one kind of number,
/// `R`, whose plain form is the same type of `f64`s; or, for a type of no kind of number,
/// whose values hold no real number, for every `R`, each value its own plain form.
///
/// Each field that holds an `R`, as itself or within another type, is lifted at the path
/// serde writes it at: under its name, its `#[serde(rename)]`, or the name the container's
/// `#[serde(rename_all)]` gives it; within a list or a tuple, under its index; within an
/// enum, under its variant's name, unless the enum is `#[serde(untagged)]`; and through a
/// newtype, a newtype variant or a `#[serde(transparent)]` struct, at the path of what
/// holds it. Every other field is cloned.
#[proc_macro_derive(Lift)]
pub fn lift(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    derived(&input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Declares a geometry-binned parameter struct, its `Default` implementation, its companion
/// binned struct, and the `binned` method that sums each binned parameter with its length,
/// width, and area dependences.
///
/// Fields are written in the exact order they appear on the parameter struct, each with its
/// default expression after `=`. A field prefixed with `binned` expands into four consecutive
/// fields on the parameter struct: itself and its `l`, `w`, and `p` dependences (escaped as
/// raw identifiers such as `r#pub` when a prefixed name is a Rust keyword), each defaulting to
/// `R::constant(0.0)`, together with a field on the companion binned struct and its evaluation
/// in `binned`.
///
/// An optional binned field (`Option<R>`) either remains `Option<R>` on the binned struct
/// (mapped when present) or carries a fallback expression after `=>` (unwrapped before adding
/// the dependences, yielding `R` on the binned struct). Prefixing a field with
/// `binned(card_only)` expands the four fields on the parameter struct without adding a field
/// to the companion binned struct.
#[proc_macro]
pub fn binned_parameters(input: TokenStream) -> TokenStream {
    let specification = parse_macro_input!(input as BinnedParameters);
    expand_binned_parameters(&specification)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// The implementation of `Lift` for `input`.
fn derived(input: &DeriveInput) -> Result<Tokens> {
    if matches!(input.data, Data::Union(_)) {
        return Err(Error::new_spanned(
            input,
            "Lift is derived for structs and enums",
        ));
    }
    let mut parameters = input.generics.params.iter();
    let number = match (parameters.next(), parameters.next()) {
        (None, _) => return Ok(holding_none(&input.ident)),
        (Some(GenericParam::Type(number)), None) => number,
        _ => {
            return Err(Error::new_spanned(
                &input.generics,
                "Lift is derived for a type generic over one kind of number, or over nothing",
            ));
        }
    };
    let number = &number.ident;
    let name = &input.ident;
    let container = Naming::of(&input.attrs)?;
    let body = match &input.data {
        Data::Struct(data) => {
            let transparent = container.transparent || is_newtype(&data.fields);
            let (pattern, construction) = lifted_fields(
                &data.fields,
                number,
                container.rename_all,
                transparent,
                &quote!(Self),
            )?;
            quote! {
                let #name #pattern = plain;
                #construction
            }
        }
        Data::Enum(data) => {
            let mut arms = Vec::with_capacity(data.variants.len());
            for variant in &data.variants {
                let naming = Naming::of(&variant.attrs)?;
                let ident = &variant.ident;
                let key = naming.rename.unwrap_or_else(|| {
                    container
                        .rename_all
                        .map_or_else(|| unraw(ident), |case| case.variant(&unraw(ident)))
                });
                let (pattern, construction) = lifted_fields(
                    &variant.fields,
                    number,
                    naming.rename_all,
                    is_newtype(&variant.fields),
                    &quote!(Self::#ident),
                )?;
                let construction = if container.untagged || variant.fields.is_empty() {
                    construction
                } else {
                    quote!({
                        let path = &path.key(#key);
                        #construction
                    })
                };
                arms.push(quote!(#name::#ident #pattern => #construction));
            }
            quote! {
                match plain {
                    #(#arms,)*
                }
            }
        }
        Data::Union(_) => {
            return Err(Error::new_spanned(
                input,
                "Lift is derived for structs and enums",
            ));
        }
    };
    let (implementation, generics, conditions) = input.generics.split_for_impl();
    Ok(quote! {
        impl #implementation ::jolty_models::Lift<#number> for #name #generics #conditions {
            type Plain = #name<f64>;

            fn lift<Lifting: FnMut(&::jolty_models::ParameterPath<'_>, f64) -> #number>(
                plain: &Self::Plain,
                path: &::jolty_models::ParameterPath<'_>,
                number: &mut Lifting,
            ) -> Self {
                #body
            }
        }
    })
}

/// The implementation of `Lift` for `name`, a type of no kind of number, whose values hold
/// no real number to lift: each is its own plain form.
fn holding_none(name: &Ident) -> Tokens {
    quote! {
        impl<Number> ::jolty_models::Lift<Number> for #name {
            type Plain = Self;

            fn lift<Lifting: FnMut(&::jolty_models::ParameterPath<'_>, f64) -> Number>(
                plain: &Self,
                _: &::jolty_models::ParameterPath<'_>,
                _: &mut Lifting,
            ) -> Self {
                ::core::clone::Clone::clone(plain)
            }
        }
    }
}

/// The pattern that binds each of `fields` of the plain value, and the expression that
/// makes the value, `constructor`, of them lifted, each at its key under the path: its
/// name, as `case` renames it, or its index, unless the fields are `transparent`.
fn lifted_fields(
    fields: &Fields,
    number: &Ident,
    case: Option<Case>,
    transparent: bool,
    constructor: &Tokens,
) -> Result<(Tokens, Tokens)> {
    let mut bindings = Vec::with_capacity(fields.len());
    let mut values = Vec::with_capacity(fields.len());
    for (index, field) in fields.iter().enumerate() {
        let naming = Naming::of(&field.attrs)?;
        let binding = Ident::new(&format!("field{index}"), Span::call_site());
        let within = if let Some(ident) = &field.ident {
            let key = naming.rename.unwrap_or_else(|| {
                case.map_or_else(|| unraw(ident), |case| case.field(&unraw(ident)))
            });
            quote!(&path.key(#key))
        } else {
            quote!(&path.index(#index))
        };
        let within = if transparent { quote!(path) } else { within };
        let value = if holds(&field.ty, number) {
            let field_type = &field.ty;
            quote!(<#field_type as ::jolty_models::Lift<#number>>::lift(#binding, #within, number))
        } else {
            quote!(::core::clone::Clone::clone(#binding))
        };
        if let Some(ident) = &field.ident {
            bindings.push(quote!(#ident: #binding));
            values.push(quote!(#ident: #value));
        } else {
            bindings.push(quote!(#binding));
            values.push(value);
        }
    }
    Ok(match fields {
        Fields::Named(_) => (
            quote!({ #(#bindings),* }),
            quote!(#constructor { #(#values),* }),
        ),
        Fields::Unnamed(_) => (
            quote!(( #(#bindings),* )),
            quote!(#constructor( #(#values),* )),
        ),
        Fields::Unit => (Tokens::new(), constructor.clone()),
    })
}

/// Whether `fields` are a newtype's, which serde writes as the one value it holds.
fn is_newtype(fields: &Fields) -> bool {
    matches!(fields, Fields::Unnamed(unnamed) if unnamed.unnamed.len() == 1)
}

/// Returns whether `field_type` references the type parameter `number` as itself or within
/// a compound type, skipping const generic expressions, array lengths, module and
/// associated-type segment names, and parameter names.
fn holds(field_type: &Type, number: &Ident) -> bool {
    match field_type {
        Type::Path(type_path) => {
            let head_holds = match &type_path.qself {
                Some(qself) => holds(&qself.ty, number),
                None => {
                    type_path.path.leading_colon.is_none()
                        && type_path.path.segments.first().is_some_and(|first| {
                            first.ident == *number && first.arguments.is_none()
                        })
                }
            };
            head_holds || path_arguments_hold(&type_path.path, number)
        }
        Type::Array(array) => holds(&array.elem, number),
        Type::Slice(slice) => holds(&slice.elem, number),
        Type::Tuple(tuple) => tuple.elems.iter().any(|elem| holds(elem, number)),
        Type::Reference(reference) => holds(&reference.elem, number),
        Type::Ptr(pointer) => holds(&pointer.elem, number),
        Type::Paren(paren) => holds(&paren.elem, number),
        Type::Group(group) => holds(&group.elem, number),
        Type::BareFn(function) => {
            function.inputs.iter().any(|arg| holds(&arg.ty, number))
                || return_type_holds(&function.output, number)
        }
        Type::ImplTrait(impl_trait) => bounds_hold(&impl_trait.bounds, number),
        Type::TraitObject(trait_object) => bounds_hold(&trait_object.bounds, number),
        // Infer, Never, Macro, Verbatim, and future non-exhaustive variants hold no type
        // parameter directly.
        _ => false,
    }
}

/// Returns whether any generic argument in any segment of `path` references `number`.
fn path_arguments_hold(path: &Path, number: &Ident) -> bool {
    path.segments
        .iter()
        .any(|segment| match &segment.arguments {
            PathArguments::None => false,
            PathArguments::AngleBracketed(bracketed) => bracketed
                .args
                .iter()
                .any(|arg| generic_argument_holds(arg, number)),
            PathArguments::Parenthesized(parenthesized) => {
                parenthesized
                    .inputs
                    .iter()
                    .any(|input| holds(input, number))
                    || return_type_holds(&parenthesized.output, number)
            }
        })
}

/// Returns whether a generic argument references `number` in a type position, skipping
/// lifetimes, const expressions, and associated item names.
fn generic_argument_holds(argument: &GenericArgument, number: &Ident) -> bool {
    match argument {
        GenericArgument::Type(ty) => holds(ty, number),
        GenericArgument::AssocType(assoc) => {
            holds(&assoc.ty, number)
                || assoc.generics.as_ref().is_some_and(|generics| {
                    generics
                        .args
                        .iter()
                        .any(|inner| generic_argument_holds(inner, number))
                })
        }
        GenericArgument::Constraint(constraint) => {
            bounds_hold(&constraint.bounds, number)
                || constraint.generics.as_ref().is_some_and(|generics| {
                    generics
                        .args
                        .iter()
                        .any(|inner| generic_argument_holds(inner, number))
                })
        }
        GenericArgument::AssocConst(assoc) => assoc.generics.as_ref().is_some_and(|generics| {
            generics
                .args
                .iter()
                .any(|inner| generic_argument_holds(inner, number))
        }),
        _ => false,
    }
}

/// Returns whether any trait bound in `bounds` references `number` in its generic arguments.
fn bounds_hold(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, Token![+]>,
    number: &Ident,
) -> bool {
    bounds.iter().any(|bound| {
        matches!(
            bound,
            TypeParamBound::Trait(trait_bound) if path_arguments_hold(&trait_bound.path, number)
        )
    })
}

/// Returns whether a function's return type references `number`.
fn return_type_holds(output: &ReturnType, number: &Ident) -> bool {
    match output {
        ReturnType::Default => false,
        ReturnType::Type(_, ty) => holds(ty, number),
    }
}

/// The name serde gives `ident`: itself, without the prefix of a raw identifier.
fn unraw(ident: &Ident) -> String {
    let name = ident.to_string();
    match name.strip_prefix("r#") {
        Some(stripped) => stripped.to_owned(),
        None => name,
    }
}

/// What a type's, a variant's or a field's `#[serde(...)]` attributes say of the path it
/// is written at.
#[derive(Default)]
struct Naming {
    /// Its `rename`.
    rename: Option<String>,
    /// Its `rename_all`.
    rename_all: Option<Case>,
    /// Whether it is `untagged`.
    untagged: bool,
    /// Whether it is `transparent`.
    transparent: bool,
}

impl Naming {
    /// What `attributes` say, failing on any that writes a value elsewhere than its path,
    /// as `flatten` or `with` do, which `Lift` does not follow.
    fn of(attributes: &[Attribute]) -> Result<Self> {
        let mut naming = Self::default();
        for attribute in attributes
            .iter()
            .filter(|attribute| attribute.path().is_ident("serde"))
        {
            attribute.parse_nested_meta(|meta| {
                let key = meta
                    .path
                    .get_ident()
                    .map_or_else(String::new, ToString::to_string);
                match key.as_str() {
                    "rename" => naming.rename = Some(meta.value()?.parse::<LitStr>()?.value()),
                    "rename_all" => naming.rename_all = Some(Case::read(&meta)?),
                    "untagged" => naming.untagged = true,
                    "transparent" => naming.transparent = true,
                    "flatten" | "skip" | "skip_serializing" | "skip_deserializing" | "tag"
                    | "content" | "rename_all_fields" | "remote" | "with" | "serialize_with"
                    | "into" | "getter" => {
                        return Err(meta.error(format!("Lift does not follow `serde({key})`")));
                    }
                    _ => skip(&meta)?,
                }
                Ok(())
            })?;
        }
        Ok(naming)
    }
}

/// Reads past an attribute `Lift` has no use for: bare, given a value, or a list.
fn skip(meta: &ParseNestedMeta<'_>) -> Result<()> {
    if meta.input.peek(Token![=]) {
        meta.value()?.parse::<syn::Expr>()?;
    } else if meta.input.peek(syn::token::Paren) {
        meta.parse_nested_meta(|inner| skip(&inner))?;
    }
    Ok(())
}

/// A case `rename_all` names fields and variants in.
#[derive(Clone, Copy)]
enum Case {
    /// `lowercase`.
    Lower,
    /// `snake_case`.
    Snake,
}

impl Case {
    /// The case `meta`, a `rename_all`, names.
    fn read(meta: &ParseNestedMeta<'_>) -> Result<Self> {
        let case = meta.value()?.parse::<LitStr>()?;
        match case.value().as_str() {
            "lowercase" => Ok(Self::Lower),
            "snake_case" => Ok(Self::Snake),
            _ => Err(Error::new(
                case.span(),
                "Lift follows `rename_all` in \"lowercase\" and \"snake_case\" alone",
            )),
        }
    }

    /// The name of the variant `name` in this case, as serde gives it.
    fn variant(self, name: &str) -> String {
        match self {
            Self::Lower => name.to_ascii_lowercase(),
            Self::Snake => {
                let mut snake = String::with_capacity(name.len() + 4);
                for (index, character) in name.char_indices() {
                    if character.is_uppercase() && index > 0 {
                        snake.push('_');
                    }
                    snake.push(character.to_ascii_lowercase());
                }
                snake
            }
        }
    }

    /// The name of the field `name` in this case, as `serde_derive` gives it: both
    /// `RenameRule::LowerCase` and `RenameRule::SnakeCase` return `field.to_owned()` in
    /// `apply_to_field`, even for mixed-case field identifiers.
    fn field(self, name: &str) -> String {
        match self {
            Self::Lower | Self::Snake => name.to_owned(),
        }
    }
}

/// Parsed input of [`binned_parameters`](macro@binned_parameters).
struct BinnedParameters {
    /// Attributes on the parameter struct, including doc comments and derives.
    parameters_attributes: Vec<Attribute>,
    /// Visibility of the parameter struct.
    parameters_visibility: Visibility,
    /// Name of the parameter struct.
    parameters_name: Ident,
    /// Generics of the parameter struct, holding a single number type parameter.
    parameters_generics: Generics,
    /// Declared fields of the parameter struct, in exact declaration order.
    fields: Vec<ParameterField>,
    /// Attributes on the companion binned struct, including doc comments and derives.
    binned_attributes: Vec<Attribute>,
    /// Visibility of the companion binned struct, its fields, and `binned`.
    binned_visibility: Visibility,
    /// Name of the companion binned struct.
    binned_name: Ident,
    /// Generics of the companion binned struct, holding the same number type parameter.
    binned_generics: Generics,
}

/// How a declared field participates in geometry binning.
#[derive(Clone, Copy)]
enum BinningMode {
    /// A single field on the parameter struct, not binned.
    Plain,
    /// Expands to four fields on the parameter struct and one evaluated field on the binned
    /// struct.
    Binned,
    /// Expands to four fields on the parameter struct without adding a field to the binned
    /// struct.
    CardOnly,
}

/// Whether a binned field has type `R` or `Option<R>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinnedShape {
    /// Bare number type parameter, `R`.
    Required,
    /// Optional number type parameter, `Option<R>`.
    Optional,
}

/// One field declaration inside [`BinnedParameters`].
struct ParameterField {
    /// Outer attributes (doc comments, serde attributes, and similar) on the base field.
    attributes: Vec<Attribute>,
    /// Whether and how the field expands into binned dependences.
    mode: BinningMode,
    /// Visibility of the base field and any generated dependence fields.
    visibility: Visibility,
    /// Identifier of the base field.
    name: Ident,
    /// Declared type of the base field.
    field_type: Type,
    /// Default value expression for `Default::default`.
    default_value: Tokens,
    /// Optional fallback expression for an optional binned field, with the span of `=>`.
    fallback: Option<(Span, Tokens)>,
}

impl Parse for BinnedParameters {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let parameters_attributes = input.call(Attribute::parse_outer)?;
        let parameters_visibility: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let parameters_name: Ident = input.parse()?;
        let mut parameters_generics: Generics = input.parse()?;
        parameters_generics.where_clause = input.parse()?;

        let body;
        braced!(body in input);
        let mut fields = Vec::new();
        while !body.is_empty() {
            fields.push(body.parse::<ParameterField>()?);
            if !body.is_empty() {
                body.parse::<Token![,]>()?;
            }
        }

        let binned_attributes = input.call(Attribute::parse_outer)?;
        let binned_visibility: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let binned_name: Ident = input.parse()?;
        let mut binned_generics: Generics = input.parse()?;
        binned_generics.where_clause = input.parse()?;
        input.parse::<Token![;]>()?;

        if !input.is_empty() {
            return Err(input.error("unexpected tokens after binned struct declaration"));
        }

        Ok(Self {
            parameters_attributes,
            parameters_visibility,
            parameters_name,
            parameters_generics,
            fields,
            binned_attributes,
            binned_visibility,
            binned_name,
            binned_generics,
        })
    }
}

impl Parse for ParameterField {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let attributes = input.call(Attribute::parse_outer)?;
        let mode = parse_binning_mode(input)?;
        let visibility: Visibility = input.parse()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let field_type: Type = input.parse()?;
        input.parse::<Token![=]>()?;
        let default_value = parse_expression_tokens(input, true)?;
        let fallback = if input.peek(Token![=>]) {
            let arrow_span = input.span();
            input.parse::<Token![=>]>()?;
            let expression = parse_expression_tokens(input, false)?;
            Some((arrow_span, expression))
        } else {
            None
        };
        Ok(Self {
            attributes,
            mode,
            visibility,
            name,
            field_type,
            default_value,
            fallback,
        })
    }
}

/// Parses an optional `binned` or `binned(card_only)` modifier before a field's visibility.
fn parse_binning_mode(input: ParseStream<'_>) -> Result<BinningMode> {
    let ahead = input.fork();
    let Ok(ident) = ahead.parse::<Ident>() else {
        return Ok(BinningMode::Plain);
    };
    if ident != "binned" || ahead.peek(Token![:]) {
        return Ok(BinningMode::Plain);
    }
    input.parse::<Ident>()?;
    if input.peek(syn::token::Paren) {
        let content;
        parenthesized!(content in input);
        let option: Ident = content.parse()?;
        if option != "card_only" {
            return Err(Error::new_spanned(option, "expected `card_only`"));
        }
        if !content.is_empty() {
            return Err(content.error("unexpected tokens after `card_only`"));
        }
        Ok(BinningMode::CardOnly)
    } else {
        Ok(BinningMode::Binned)
    }
}

/// Collects expression tokens until a top-level `,`, `=>` (when `stop_at_arrow` is true), or
/// the end of `input`.
fn parse_expression_tokens(input: ParseStream<'_>, stop_at_arrow: bool) -> Result<Tokens> {
    let start = input.span();
    let mut tokens = Tokens::new();
    while !input.is_empty()
        && !input.peek(Token![,])
        && !(stop_at_arrow && input.peek(Token![=>]))
    {
        let tree: TokenTree = input.parse()?;
        tokens.extend([tree]);
    }
    if tokens.is_empty() {
        return Err(Error::new(start, "expected an expression"));
    }
    Ok(tokens)
}

/// Extracts the single number type parameter from `generics`.
fn single_type_parameter(generics: &Generics) -> Result<&Ident> {
    let mut parameters = generics.params.iter();
    match (parameters.next(), parameters.next()) {
        (Some(GenericParam::Type(number)), None) => Ok(&number.ident),
        _ => Err(Error::new_spanned(
            generics,
            "binned_parameters requires a struct generic over one kind of number",
        )),
    }
}

/// Returns the single unprefixed path segment of `field_type`, if it is a plain path.
fn single_segment(field_type: &Type) -> Option<&syn::PathSegment> {
    let Type::Path(type_path) = field_type else {
        return None;
    };
    if type_path.qself.is_some()
        || type_path.path.leading_colon.is_some()
        || type_path.path.segments.len() != 1
    {
        return None;
    }
    type_path.path.segments.first()
}

/// Returns whether `field_type` is `number` directly or `Option<number>`.
fn classify_binned_type(field_type: &Type, number: &Ident) -> Option<BinnedShape> {
    let segment = single_segment(field_type)?;
    if segment.ident == *number && segment.arguments.is_none() {
        return Some(BinnedShape::Required);
    }
    if segment.ident == "Option"
        && let PathArguments::AngleBracketed(arguments) = &segment.arguments
        && arguments.args.len() == 1
        && let Some(GenericArgument::Type(inner)) = arguments.args.first()
        && let Some(inner_segment) = single_segment(inner)
        && inner_segment.ident == *number
        && inner_segment.arguments.is_none()
    {
        return Some(BinnedShape::Optional);
    }
    None
}

/// Returns whether a dependence name prefixed with `l`, `w`, or `p` is a Rust keyword that
/// requires raw-identifier escaping.
fn is_dependence_keyword(name: &str) -> bool {
    matches!(name, "let" | "loop" | "priv" | "pub" | "where" | "while")
}

/// Builds the `prefix`-prefixed dependence unraw name and identifier for `raw`, escaping Rust
/// keywords as raw identifiers (such as `r#pub` for `ub`).
fn prefixed_ident(prefix: char, raw: &str, span: Span) -> (String, Ident) {
    let name = format!("{prefix}{raw}");
    let ident = if is_dependence_keyword(&name) {
        Ident::new_raw(&name, span)
    } else {
        Ident::new(&name, span)
    };
    (name, ident)
}

/// The unraw base name and generated `l`, `w`, and `p` dependence names and identifiers for a
/// binned field.
struct Dependences {
    /// Unraw base name, such as `"ub"`.
    base: String,
    /// Length dependence unraw name and identifier, such as `("lub", lub)`.
    length: (String, Ident),
    /// Width dependence unraw name and identifier, such as `("wub", wub)`.
    width: (String, Ident),
    /// Area dependence unraw name and identifier, such as `("pub", r#pub)`.
    area: (String, Ident),
}

impl Dependences {
    /// Builds the length, width, and area dependence names and identifiers for `base`.
    fn for_field(base: &Ident) -> Self {
        let raw = unraw(base);
        let span = base.span();
        Self {
            length: prefixed_ident('l', &raw, span),
            width: prefixed_ident('w', &raw, span),
            area: prefixed_ident('p', &raw, span),
            base: raw,
        }
    }
}

/// Accumulated token streams for the items emitted by
/// [`binned_parameters`](macro@binned_parameters).
struct ExpandedFields {
    /// Fields of the parameter struct, including generated `l`, `w`, and `p` dependences.
    parameter_fields: Vec<Tokens>,
    /// Field initializers in `Default::default` for the parameter struct.
    default_fields: Vec<Tokens>,
    /// Fields of the companion binned struct.
    binned_fields: Vec<Tokens>,
    /// Field initializers in `binned`.
    binned_values: Vec<Tokens>,
}

impl ExpandedFields {
    /// Validates and expands all fields of `specification`.
    fn from_specification(specification: &BinnedParameters) -> Result<Self> {
        let number = single_type_parameter(&specification.parameters_generics)?;
        let binned_number = single_type_parameter(&specification.binned_generics)?;
        if binned_number != number {
            let message = format!(
                "expected the same number type parameter `{number}` as the parameter struct"
            );
            return Err(Error::new_spanned(binned_number, message));
        }
        let mut expanded = Self {
            parameter_fields: Vec::with_capacity(specification.fields.len()),
            default_fields: Vec::with_capacity(specification.fields.len()),
            binned_fields: Vec::new(),
            binned_values: Vec::new(),
        };
        let binned_visibility = &specification.binned_visibility;
        for field in &specification.fields {
            expanded.push(field, number, binned_visibility)?;
        }
        Ok(expanded)
    }

    /// Appends the tokens for one declared `field` to each emitted item.
    fn push(
        &mut self,
        field: &ParameterField,
        number: &Ident,
        binned_visibility: &Visibility,
    ) -> Result<()> {
        let attributes = &field.attributes;
        let visibility = &field.visibility;
        let name = &field.name;
        let field_type = &field.field_type;
        let default_value = &field.default_value;

        let declaration = quote!(#(#attributes)* #visibility #name: #field_type);
        self.parameter_fields.push(declaration);
        self.default_fields.push(quote!(#name: #default_value));

        if matches!(field.mode, BinningMode::Plain) {
            if let Some((span, _)) = &field.fallback {
                let message = "a fallback expression applies only to an optional binned field";
                return Err(Error::new(*span, message));
            }
            return Ok(());
        }

        let Some(shape) = classify_binned_type(field_type, number) else {
            let message = format!("a binned field must have type `{number}` or `Option<{number}>`");
            return Err(Error::new_spanned(field_type, message));
        };

        let dependences = Dependences::for_field(name);
        self.push_dependences(field, number, &dependences);

        if matches!(field.mode, BinningMode::CardOnly) {
            if let Some((span, _)) = &field.fallback {
                let message = "a fallback expression does not apply to a `binned(card_only)` field";
                return Err(Error::new(*span, message));
            }
            return Ok(());
        }

        self.push_binned_entry(field, shape, number, binned_visibility, &dependences)
    }

    /// Appends the `l`, `w`, and `p` dependence fields and zero defaults for `field`.
    fn push_dependences(
        &mut self,
        field: &ParameterField,
        number: &Ident,
        dependences: &Dependences,
    ) {
        let visibility = &field.visibility;
        let base = &dependences.base;
        let (length_name, length_ident) = &dependences.length;
        let (width_name, width_ident) = &dependences.width;
        let (area_name, area_ident) = &dependences.area;
        let length_doc = format!(
            "The length dependence of `{base}`, `{length_name}`: its coefficient of the \
             reciprocal of the effective length, in the unit of `{base}` times meters; default 0."
        );
        let width_doc = format!(
            "The width dependence of `{base}`, `{width_name}`: its coefficient of the \
             reciprocal of the effective width, in the unit of `{base}` times meters; default 0."
        );
        let area_doc = format!(
            "The area dependence of `{base}`, `{area_name}`: its coefficient of the \
             reciprocal of the effective length times the effective width, in the unit of `{base}` \
             times square meters; default 0."
        );

        self.parameter_fields.extend([
            quote!(#[doc = #length_doc] #visibility #length_ident: #number),
            quote!(#[doc = #width_doc] #visibility #width_ident: #number),
            quote!(#[doc = #area_doc] #visibility #area_ident: #number),
        ]);
        self.default_fields.extend([
            quote!(#length_ident: #number::constant(0.0)),
            quote!(#width_ident: #number::constant(0.0)),
            quote!(#area_ident: #number::constant(0.0)),
        ]);
    }

    /// Appends the field declaration on the companion binned struct and its evaluation in
    /// `binned`.
    fn push_binned_entry(
        &mut self,
        field: &ParameterField,
        shape: BinnedShape,
        number: &Ident,
        binned_visibility: &Visibility,
        dependences: &Dependences,
    ) -> Result<()> {
        let name = &field.name;
        let field_type = &field.field_type;
        let base = &dependences.base;
        let length_ident = &dependences.length.1;
        let width_ident = &dependences.width.1;
        let area_ident = &dependences.area.1;
        let binned_doc = format!("The binned value of `{base}`.");

        let (binned_type, value) = match (shape, &field.fallback) {
            (BinnedShape::Required, Some((span, _))) => {
                let message = format!(
                    "a fallback expression applies only to an `Option<{number}>` field"
                );
                return Err(Error::new(*span, message));
            }
            (BinnedShape::Required, None) => (
                quote!(#number),
                quote! {
                    #name: self.#name
                        + self.#length_ident * inverse_length
                        + self.#width_ident * inverse_width
                        + self.#area_ident * inverse_area
                },
            ),
            (BinnedShape::Optional, Some((_, fallback))) => (
                quote!(#number),
                quote! {
                    #name: self.#name.unwrap_or(#fallback)
                        + self.#length_ident * inverse_length
                        + self.#width_ident * inverse_width
                        + self.#area_ident * inverse_area
                },
            ),
            (BinnedShape::Optional, None) => (
                quote!(#field_type),
                quote! {
                    #name: self.#name.map(|value| {
                        value
                            + self.#length_ident * inverse_length
                            + self.#width_ident * inverse_width
                            + self.#area_ident * inverse_area
                    })
                },
            ),
        };
        let declaration = quote!(#[doc = #binned_doc] #binned_visibility #name: #binned_type);
        self.binned_fields.push(declaration);
        self.binned_values.push(value);
        Ok(())
    }
}

/// Expands [`BinnedParameters`] into the parameter struct, its `Default` implementation, the
/// companion binned struct, and the `binned` method.
fn expand_binned_parameters(specification: &BinnedParameters) -> Result<Tokens> {
    let number = single_type_parameter(&specification.parameters_generics)?;
    let expanded = ExpandedFields::from_specification(specification)?;

    let parameters_attributes = &specification.parameters_attributes;
    let parameters_visibility = &specification.parameters_visibility;
    let parameters_name = &specification.parameters_name;
    let parameters_generics = &specification.parameters_generics;
    let (impl_generics, type_generics, where_clause) = parameters_generics.split_for_impl();

    let binned_attributes = &specification.binned_attributes;
    let binned_visibility = &specification.binned_visibility;
    let binned_name = &specification.binned_name;
    let binned_generics = &specification.binned_generics;
    let (_, binned_type_generics, binned_where) = binned_generics.split_for_impl();

    let parameter_fields = &expanded.parameter_fields;
    let default_fields = &expanded.default_fields;
    let binned_fields = &expanded.binned_fields;
    let binned_values = &expanded.binned_values;

    Ok(quote! {
        #(#parameters_attributes)*
        #parameters_visibility struct #parameters_name #parameters_generics #where_clause {
            #(#parameter_fields,)*
        }

        impl #impl_generics ::core::default::Default
            for #parameters_name #type_generics #where_clause
        {
            fn default() -> Self {
                Self {
                    #(#default_fields,)*
                }
            }
        }

        #(#binned_attributes)*
        #binned_visibility struct #binned_name #binned_generics #binned_where {
            #(#binned_fields,)*
        }

        impl #impl_generics #parameters_name #type_generics #where_clause {
            /// The binned parameters at the reciprocals of the effective length and width, and
            /// of their product.
            #binned_visibility fn binned(
                &self,
                [inverse_length, inverse_width, inverse_area]: [#number; 3],
            ) -> #binned_name #binned_type_generics {
                #binned_name {
                    #(#binned_values,)*
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unraw_strips_raw_identifier_prefix() {
        let raw_type: Ident = syn::parse_str("r#type").unwrap();
        let raw_match: Ident = syn::parse_str("r#match").unwrap();
        let plain = Ident::new("resistance", Span::call_site());
        let prefix_like = Ident::new("raw_value", Span::call_site());
        assert_eq!(unraw(&raw_type), "type");
        assert_eq!(unraw(&raw_match), "match");
        assert_eq!(unraw(&plain), "resistance");
        assert_eq!(unraw(&prefix_like), "raw_value");
    }

    #[test]
    fn case_variant_converts_variant_names_and_preserves_fields() {
        assert_eq!(Case::Lower.variant("PiecewiseLinear"), "piecewiselinear");
        assert_eq!(Case::Lower.variant("DC"), "dc");
        assert_eq!(Case::Snake.variant("Pulse"), "pulse");
        assert_eq!(Case::Snake.variant("PiecewiseLinear"), "piecewise_linear");
        assert_eq!(Case::Snake.variant("SingleEnded"), "single_ended");
        assert_eq!(Case::Lower.field("camelCase"), "camelCase");
        assert_eq!(Case::Snake.field("camelCase"), "camelCase");
    }

    #[test]
    fn holds_walks_types_without_false_positives() {
        let number = Ident::new("R", Span::call_site());
        for present in [
            "R",
            "R::Scalar",
            "Vec<R>",
            "Option<Vec<(f64, R)>>",
            "[R; 4]",
            "[R]",
            "&'a R",
            "*const R",
            "(R)",
            "<R as Trait>::Item",
            "foo::Bar<R>",
            "dyn Iterator<Item = R>",
            "impl Fn(R) -> f64",
            "fn(R) -> i32",
            "fn(i32) -> R",
            "R<R>",
        ] {
            let parsed: Type = syn::parse_str(present).unwrap();
            assert!(holds(&parsed, &number), "expected `{present}` to hold `R`");
        }
        for absent in [
            "f64",
            "String",
            "Vec<f64>",
            "[f64; R]",
            "foo::R",
            "::R",
            "crate::R",
            "Self::R",
            "<Foo as Bar>::R",
            "<Foo as R>::Item",
            "Foo<{ R }>",
            "Iterator<R = i32>",
            "Trait<R = 4>",
            "Trait<R: Clone>",
            "fn(R: i32) -> i32",
            "dyn R",
            "R<i32>",
        ] {
            let parsed: Type = syn::parse_str(absent).unwrap();
            assert!(
                !holds(&parsed, &number),
                "expected `{absent}` not to hold `R`"
            );
        }
    }

    #[test]
    fn derived_rejects_unions_with_or_without_generics() {
        for source in [
            "union Plain { a: u32 }",
            "union Generic<R> { a: core::mem::ManuallyDrop<R> }",
        ] {
            let input: DeriveInput = syn::parse_str(source).unwrap();
            let error = derived(&input).unwrap_err();
            assert_eq!(error.to_string(), "Lift is derived for structs and enums");
        }

        let plain_struct: DeriveInput = syn::parse_str("struct Plain { a: u32 }").unwrap();
        assert!(derived(&plain_struct).is_ok());

        let two_params: DeriveInput = syn::parse_str("struct Pair<A, B> { a: A, b: B }").unwrap();
        let error = derived(&two_params).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Lift is derived for a type generic over one kind of number, or over nothing"
        );
    }

    fn parsed_fields(tokens: &[Tokens]) -> Vec<syn::Field> {
        let stream = quote!(pub struct Check<R> { #(#tokens,)* });
        let input: DeriveInput = syn::parse2(stream).unwrap();
        let Data::Struct(data) = input.data else {
            panic!("expected a struct");
        };
        data.fields.into_iter().collect()
    }

    fn doc_of(attributes: &[Attribute]) -> Option<String> {
        for attribute in attributes {
            if attribute.path().is_ident("doc")
                && let syn::Meta::NameValue(name_value) = &attribute.meta
                && let syn::Expr::Lit(literal) = &name_value.value
                && let syn::Lit::Str(text) = &literal.lit
            {
                return Some(text.value());
            }
        }
        None
    }

    fn compact(tokens: &Tokens) -> String {
        tokens.to_string().replace(' ', "")
    }

    #[test]
    fn prefixed_ident_escapes_rust_keywords() {
        let span = Span::call_site();
        for (prefix, raw, expected_name, expected_ident) in [
            ('l', "ub", "lub", "lub"),
            ('w', "ub", "wub", "wub"),
            ('p', "ub", "pub", "r#pub"),
            ('l', "et", "let", "r#let"),
            ('w', "here", "where", "r#where"),
        ] {
            let (name, ident) = prefixed_ident(prefix, raw, span);
            assert_eq!(name, expected_name);
            assert_eq!(ident.to_string(), expected_ident);
        }

        let raw_base: Ident = syn::parse_str("r#type").unwrap();
        let dependences = Dependences::for_field(&raw_base);
        assert_eq!(dependences.base, "type");
        assert_eq!(dependences.length.1.to_string(), "ltype");
        assert_eq!(dependences.width.1.to_string(), "wtype");
        assert_eq!(dependences.area.1.to_string(), "ptype");
    }

    #[test]
    fn classify_binned_type_recognizes_number_and_option() {
        let number = Ident::new("R", Span::call_site());
        let required: Type = syn::parse_str("R").unwrap();
        let optional: Type = syn::parse_str("Option<R>").unwrap();
        assert_eq!(
            classify_binned_type(&required, &number),
            Some(BinnedShape::Required)
        );
        assert_eq!(
            classify_binned_type(&optional, &number),
            Some(BinnedShape::Optional)
        );

        for rejected in [
            "f64",
            "bool",
            "Option<f64>",
            "Vec<R>",
            "crate::R",
            "Option<Option<R>>",
        ] {
            let parsed: Type = syn::parse_str(rejected).unwrap();
            assert_eq!(classify_binned_type(&parsed, &number), None);
        }
    }

    #[test]
    fn binned_parameters_expands_fields_in_declaration_order() {
        let source = r#"
            /// Sample parameters.
            #[derive(Debug, Clone)]
            pub struct SampleParameters<R: Real = f64> {
                /// Channel selector.
                #[serde(rename = "ch")]
                pub channel: Channel = Channel::N,
                /// Plain private field named binned.
                binned: f64 = 0.0,
                /// Second-order mobility degradation.
                binned pub ub: R = R::constant(5.87e-19),
                /// Surface doping with fallback.
                binned pub nch: Option<R> = None => R::constant(1.7e23),
                /// Body effect deferred to prepare.
                binned pub gamma1: Option<R> = None,
                /// Card-only binned field.
                binned(card_only) pub eu: Option<R> = None,
                /// Trailing plain field.
                pub tail: R = R::constant(1.0),
            }

            /// Sample binned values.
            #[derive(Debug, Clone, Copy)]
            pub(super) struct Binned<R: Real>;
        "#;
        let specification: BinnedParameters = syn::parse_str(source).unwrap();
        let expanded = ExpandedFields::from_specification(&specification).unwrap();
        assert!(expand_binned_parameters(&specification).is_ok());

        let fields = parsed_fields(&expanded.parameter_fields);
        let names: Vec<String> = fields
            .iter()
            .map(|field| field.ident.as_ref().unwrap().to_string())
            .collect();
        let expected_names = [
            "channel",
            "binned",
            "ub",
            "lub",
            "wub",
            "r#pub",
            "nch",
            "lnch",
            "wnch",
            "pnch",
            "gamma1",
            "lgamma1",
            "wgamma1",
            "pgamma1",
            "eu",
            "leu",
            "weu",
            "peu",
            "tail",
        ];
        assert_eq!(names, expected_names);
        let has_serde = fields[0]
            .attrs
            .iter()
            .any(|attribute| attribute.path().is_ident("serde"));
        assert!(has_serde);
        let expected_doc = "The area dependence of `ub`, `pub`: its coefficient of the reciprocal \
                            of the effective length times the effective width, in the unit of `ub` \
                            times square meters; default 0.";
        assert_eq!(doc_of(&fields[5].attrs).as_deref(), Some(expected_doc));
        let pub_default = compact(&expanded.default_fields[5]);
        assert_eq!(pub_default, "r#pub:R::constant(0.0)");

        let binned = parsed_fields(&expanded.binned_fields);
        let binned_names: Vec<String> = binned
            .iter()
            .map(|field| field.ident.as_ref().unwrap().to_string())
            .collect();
        assert_eq!(binned_names, ["ub", "nch", "gamma1"]);
        let binned_types: Vec<String> = binned
            .iter()
            .map(|field| {
                let field_type = &field.ty;
                compact(&quote!(#field_type))
            })
            .collect();
        assert_eq!(binned_types, ["R", "R", "Option<R>"]);
    }

    #[test]
    fn binned_parameters_preserves_exact_binning_formulas() {
        let source = r"
            pub struct Sample<R: Real = f64> {
                /// Required binned field.
                binned pub ub: R = R::constant(1.0),
                /// Optional with fallback.
                binned pub u0: Option<R> = None
                    => R::constant(by_channel(self.channel, 0.067, 0.025)),
                /// Optional deferred.
                binned pub gamma1: Option<R> = None,
            }

            pub(super) struct Binned<R: Real>;
        ";
        let specification: BinnedParameters = syn::parse_str(source).unwrap();
        let expanded = ExpandedFields::from_specification(&specification).unwrap();
        let values: Vec<String> = expanded.binned_values.iter().map(compact).collect();
        assert_eq!(
            values[0],
            "ub:self.ub+self.lub*inverse_length+self.wub*inverse_width+self.r#pub*inverse_area"
        );
        assert_eq!(
            values[1],
            "u0:self.u0.unwrap_or(R::constant(by_channel(self.channel,0.067,0.025)))\
             +self.lu0*inverse_length+self.wu0*inverse_width+self.pu0*inverse_area"
        );
        assert_eq!(
            values[2],
            "gamma1:self.gamma1.map(|value|{value+self.lgamma1*inverse_length\
             +self.wgamma1*inverse_width+self.pgamma1*inverse_area})"
        );
    }

    #[test]
    fn binned_parameters_rejects_invalid_declarations() {
        let parse_errors = [
            (
                "pub struct P<R: Real> { binned(other) pub x: R = 0.0, } struct B<R: Real>;",
                "expected `card_only`",
            ),
            (
                "pub struct P<R: Real> { pub x: R = , } struct B<R: Real>;",
                "expected an expression",
            ),
            (
                "pub struct P<R: Real> { binned pub x: Option<R> = None => , } struct B<R: Real>;",
                "expected an expression",
            ),
        ];
        for (source, expected) in parse_errors {
            let error = syn::parse_str::<BinnedParameters>(source).err().unwrap();
            assert_eq!(error.to_string(), expected);
        }

        let expand_errors = [
            (
                "pub struct P<R: Real> { binned pub x: f64 = 0.0, } struct B<R: Real>;",
                "a binned field must have type `R` or `Option<R>`",
            ),
            (
                "pub struct P<R: Real> { pub x: Option<R> = None => 1.0, } struct B<R: Real>;",
                "a fallback expression applies only to an optional binned field",
            ),
            (
                "pub struct P<R: Real> { binned(card_only) pub x: Option<R> = None => 1.0, } \
                 struct B<R: Real>;",
                "a fallback expression does not apply to a `binned(card_only)` field",
            ),
            (
                "pub struct P<R: Real> { binned pub x: R = 0.0 => 1.0, } struct B<R: Real>;",
                "a fallback expression applies only to an `Option<R>` field",
            ),
            (
                "pub struct P<R: Real> { pub x: R = 0.0, } struct B<T: Real>;",
                "expected the same number type parameter `R` as the parameter struct",
            ),
            (
                "pub struct P<A, B> { pub x: A = 0.0, } struct Binned<A>;",
                "binned_parameters requires a struct generic over one kind of number",
            ),
        ];
        for (source, expected) in expand_errors {
            let specification: BinnedParameters = syn::parse_str(source).unwrap();
            let error = expand_binned_parameters(&specification).unwrap_err();
            assert_eq!(error.to_string(), expected);
        }
    }
}
