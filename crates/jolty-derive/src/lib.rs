//! The derive macros of jolty.
//!
//! [`Lift`](macro@Lift) writes `jolty_models::Lift` for a type generic over its kind of
//! number, `R`: the type of `R`s made from the same type of plain numbers, `f64`s, each
//! number by the path serde writes it at, so that a twin seeds a parameter's derivatives
//! where the parameter is named, with no detour through the parameters' written form.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Tokens, TokenTree};
use quote::{ToTokens, quote};
use syn::meta::ParseNestedMeta;
use syn::{
    Attribute, Data, DeriveInput, Error, Fields, Ident, LitStr, Result, Token, Type,
    parse_macro_input,
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
    let mut parameters = input.generics.type_params();
    let number = match (parameters.next(), parameters.next()) {
        (None, _) if input.generics.params.is_empty() => return Ok(holding_none(&input.ident)),
        (Some(number), None) => number,
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
                (number, container.rename_all),
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
                    (number, None),
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
    (number, case): (&Ident, Option<Case>),
    transparent: bool,
    constructor: &Tokens,
) -> Result<(Tokens, Tokens)> {
    let mut bindings = Vec::with_capacity(fields.len());
    let mut values = Vec::with_capacity(fields.len());
    for (index, field) in fields.iter().enumerate() {
        let naming = Naming::of(&field.attrs)?;
        let (binding, within) = match &field.ident {
            Some(ident) => {
                let key = naming.rename.unwrap_or_else(|| {
                    case.map_or_else(|| unraw(ident), |case| case.field(&unraw(ident)))
                });
                (ident.clone(), quote!(&path.key(#key)))
            }
            None => (
                Ident::new(&format!("field{index}"), Span::call_site()),
                quote!(&path.index(#index)),
            ),
        };
        let within = if transparent { quote!(path) } else { within };
        let value = if holds(&field.ty, number) {
            let ty = &field.ty;
            quote!(<#ty as ::jolty_models::Lift<#number>>::lift(#binding, #within, number))
        } else {
            quote!(::core::clone::Clone::clone(#binding))
        };
        values.push(match &field.ident {
            Some(ident) => quote!(#ident: #value),
            None => value,
        });
        bindings.push(binding);
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

/// Whether `ty` holds the kind of number `number`, as itself or within another type.
fn holds(ty: &Type, number: &Ident) -> bool {
    fn within(tokens: Tokens, number: &Ident) -> bool {
        tokens.into_iter().any(|tree| match tree {
            TokenTree::Ident(ident) => ident == *number,
            TokenTree::Group(group) => within(group.stream(), number),
            TokenTree::Punct(_) | TokenTree::Literal(_) => false,
        })
    }
    within(ty.to_token_stream(), number)
}

/// The name serde gives `ident`: itself, without the prefix of a raw identifier.
fn unraw(ident: &Ident) -> String {
    let name = ident.to_string();
    name.strip_prefix("r#")
        .map_or_else(|| name.clone(), str::to_owned)
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

    /// The name of the field `name` in this case, as serde gives it: itself, a field's
    /// name being in both cases already.
    fn field(self, name: &str) -> String {
        match self {
            Self::Lower | Self::Snake => name.to_owned(),
        }
    }
}
