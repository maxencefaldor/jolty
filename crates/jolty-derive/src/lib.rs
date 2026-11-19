//! The derive macros of jolty.
//!
//! [`Lift`](macro@Lift) writes `jolty_models::Lift` for a type generic over its kind of
//! number, `R`: the type of `R`s made from the same type of plain numbers, `f64`s, each
//! number by the path serde writes it at, so that a twin seeds a parameter's derivatives
//! where the parameter is named, with no detour through the parameters' written form.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Tokens};
use quote::quote;
use syn::meta::ParseNestedMeta;
use syn::{
    Attribute, Data, DeriveInput, Error, Fields, GenericArgument, GenericParam, Ident, LitStr,
    Path, PathArguments, Result, ReturnType, Token, Type, TypeParamBound, parse_macro_input,
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
    path.segments.iter().any(|segment| match &segment.arguments {
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
}
