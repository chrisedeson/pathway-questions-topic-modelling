
/**
 * Client
**/

import * as runtime from './runtime/library.js';
import $Types = runtime.Types // general types
import $Public = runtime.Types.Public
import $Utils = runtime.Types.Utils
import $Extensions = runtime.Types.Extensions
import $Result = runtime.Types.Result

export type PrismaPromise<T> = $Public.PrismaPromise<T>


/**
 * Model Question
 * 
 */
export type Question = $Result.DefaultSelection<Prisma.$QuestionPayload>
/**
 * Model Topic
 * 
 */
export type Topic = $Result.DefaultSelection<Prisma.$TopicPayload>
/**
 * Model TopicAssignment
 * 
 */
export type TopicAssignment = $Result.DefaultSelection<Prisma.$TopicAssignmentPayload>
/**
 * Model ClusterResult
 * 
 */
export type ClusterResult = $Result.DefaultSelection<Prisma.$ClusterResultPayload>
/**
 * Model AnalysisRun
 * 
 */
export type AnalysisRun = $Result.DefaultSelection<Prisma.$AnalysisRunPayload>
/**
 * Model SystemConfig
 * 
 */
export type SystemConfig = $Result.DefaultSelection<Prisma.$SystemConfigPayload>
/**
 * Model UserSession
 * 
 */
export type UserSession = $Result.DefaultSelection<Prisma.$UserSessionPayload>
/**
 * Model DataSync
 * 
 */
export type DataSync = $Result.DefaultSelection<Prisma.$DataSyncPayload>

/**
 * Enums
 */
export namespace $Enums {
  export const AssignmentType: {
  SIMILARITY_MATCH: 'SIMILARITY_MATCH',
  CLUSTER_DISCOVERY: 'CLUSTER_DISCOVERY',
  MANUAL_ASSIGNMENT: 'MANUAL_ASSIGNMENT'
};

export type AssignmentType = (typeof AssignmentType)[keyof typeof AssignmentType]


export const AnalysisStatus: {
  PENDING: 'PENDING',
  RUNNING: 'RUNNING',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
  CANCELLED: 'CANCELLED'
};

export type AnalysisStatus = (typeof AnalysisStatus)[keyof typeof AnalysisStatus]


export const SyncStatus: {
  PENDING: 'PENDING',
  RUNNING: 'RUNNING',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED'
};

export type SyncStatus = (typeof SyncStatus)[keyof typeof SyncStatus]


export const ConfigType: {
  STRING: 'STRING',
  NUMBER: 'NUMBER',
  BOOLEAN: 'BOOLEAN',
  JSON: 'JSON'
};

export type ConfigType = (typeof ConfigType)[keyof typeof ConfigType]

}

export type AssignmentType = $Enums.AssignmentType

export const AssignmentType: typeof $Enums.AssignmentType

export type AnalysisStatus = $Enums.AnalysisStatus

export const AnalysisStatus: typeof $Enums.AnalysisStatus

export type SyncStatus = $Enums.SyncStatus

export const SyncStatus: typeof $Enums.SyncStatus

export type ConfigType = $Enums.ConfigType

export const ConfigType: typeof $Enums.ConfigType

/**
 * ##  Prisma Client ʲˢ
 *
 * Type-safe database client for TypeScript & Node.js
 * @example
 * ```
 * const prisma = new PrismaClient()
 * // Fetch zero or more Questions
 * const questions = await prisma.question.findMany()
 * ```
 *
 *
 * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client).
 */
export class PrismaClient<
  ClientOptions extends Prisma.PrismaClientOptions = Prisma.PrismaClientOptions,
  const U = 'log' extends keyof ClientOptions ? ClientOptions['log'] extends Array<Prisma.LogLevel | Prisma.LogDefinition> ? Prisma.GetEvents<ClientOptions['log']> : never : never,
  ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs
> {
  [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['other'] }

    /**
   * ##  Prisma Client ʲˢ
   *
   * Type-safe database client for TypeScript & Node.js
   * @example
   * ```
   * const prisma = new PrismaClient()
   * // Fetch zero or more Questions
   * const questions = await prisma.question.findMany()
   * ```
   *
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client).
   */

  constructor(optionsArg ?: Prisma.Subset<ClientOptions, Prisma.PrismaClientOptions>);
  $on<V extends U>(eventType: V, callback: (event: V extends 'query' ? Prisma.QueryEvent : Prisma.LogEvent) => void): PrismaClient;

  /**
   * Connect with the database
   */
  $connect(): $Utils.JsPromise<void>;

  /**
   * Disconnect from the database
   */
  $disconnect(): $Utils.JsPromise<void>;

/**
   * Executes a prepared raw query and returns the number of affected rows.
   * @example
   * ```
   * const result = await prisma.$executeRaw`UPDATE User SET cool = ${true} WHERE email = ${'user@email.com'};`
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $executeRaw<T = unknown>(query: TemplateStringsArray | Prisma.Sql, ...values: any[]): Prisma.PrismaPromise<number>;

  /**
   * Executes a raw query and returns the number of affected rows.
   * Susceptible to SQL injections, see documentation.
   * @example
   * ```
   * const result = await prisma.$executeRawUnsafe('UPDATE User SET cool = $1 WHERE email = $2 ;', true, 'user@email.com')
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $executeRawUnsafe<T = unknown>(query: string, ...values: any[]): Prisma.PrismaPromise<number>;

  /**
   * Performs a prepared raw query and returns the `SELECT` data.
   * @example
   * ```
   * const result = await prisma.$queryRaw`SELECT * FROM User WHERE id = ${1} OR email = ${'user@email.com'};`
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $queryRaw<T = unknown>(query: TemplateStringsArray | Prisma.Sql, ...values: any[]): Prisma.PrismaPromise<T>;

  /**
   * Performs a raw query and returns the `SELECT` data.
   * Susceptible to SQL injections, see documentation.
   * @example
   * ```
   * const result = await prisma.$queryRawUnsafe('SELECT * FROM User WHERE id = $1 OR email = $2;', 1, 'user@email.com')
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $queryRawUnsafe<T = unknown>(query: string, ...values: any[]): Prisma.PrismaPromise<T>;


  /**
   * Allows the running of a sequence of read/write operations that are guaranteed to either succeed or fail as a whole.
   * @example
   * ```
   * const [george, bob, alice] = await prisma.$transaction([
   *   prisma.user.create({ data: { name: 'George' } }),
   *   prisma.user.create({ data: { name: 'Bob' } }),
   *   prisma.user.create({ data: { name: 'Alice' } }),
   * ])
   * ```
   * 
   * Read more in our [docs](https://www.prisma.io/docs/concepts/components/prisma-client/transactions).
   */
  $transaction<P extends Prisma.PrismaPromise<any>[]>(arg: [...P], options?: { isolationLevel?: Prisma.TransactionIsolationLevel }): $Utils.JsPromise<runtime.Types.Utils.UnwrapTuple<P>>

  $transaction<R>(fn: (prisma: Omit<PrismaClient, runtime.ITXClientDenyList>) => $Utils.JsPromise<R>, options?: { maxWait?: number, timeout?: number, isolationLevel?: Prisma.TransactionIsolationLevel }): $Utils.JsPromise<R>


  $extends: $Extensions.ExtendsHook<"extends", Prisma.TypeMapCb<ClientOptions>, ExtArgs, $Utils.Call<Prisma.TypeMapCb<ClientOptions>, {
    extArgs: ExtArgs
  }>>

      /**
   * `prisma.question`: Exposes CRUD operations for the **Question** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more Questions
    * const questions = await prisma.question.findMany()
    * ```
    */
  get question(): Prisma.QuestionDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.topic`: Exposes CRUD operations for the **Topic** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more Topics
    * const topics = await prisma.topic.findMany()
    * ```
    */
  get topic(): Prisma.TopicDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.topicAssignment`: Exposes CRUD operations for the **TopicAssignment** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more TopicAssignments
    * const topicAssignments = await prisma.topicAssignment.findMany()
    * ```
    */
  get topicAssignment(): Prisma.TopicAssignmentDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.clusterResult`: Exposes CRUD operations for the **ClusterResult** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more ClusterResults
    * const clusterResults = await prisma.clusterResult.findMany()
    * ```
    */
  get clusterResult(): Prisma.ClusterResultDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.analysisRun`: Exposes CRUD operations for the **AnalysisRun** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more AnalysisRuns
    * const analysisRuns = await prisma.analysisRun.findMany()
    * ```
    */
  get analysisRun(): Prisma.AnalysisRunDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.systemConfig`: Exposes CRUD operations for the **SystemConfig** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more SystemConfigs
    * const systemConfigs = await prisma.systemConfig.findMany()
    * ```
    */
  get systemConfig(): Prisma.SystemConfigDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.userSession`: Exposes CRUD operations for the **UserSession** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more UserSessions
    * const userSessions = await prisma.userSession.findMany()
    * ```
    */
  get userSession(): Prisma.UserSessionDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.dataSync`: Exposes CRUD operations for the **DataSync** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more DataSyncs
    * const dataSyncs = await prisma.dataSync.findMany()
    * ```
    */
  get dataSync(): Prisma.DataSyncDelegate<ExtArgs, ClientOptions>;
}

export namespace Prisma {
  export import DMMF = runtime.DMMF

  export type PrismaPromise<T> = $Public.PrismaPromise<T>

  /**
   * Validator
   */
  export import validator = runtime.Public.validator

  /**
   * Prisma Errors
   */
  export import PrismaClientKnownRequestError = runtime.PrismaClientKnownRequestError
  export import PrismaClientUnknownRequestError = runtime.PrismaClientUnknownRequestError
  export import PrismaClientRustPanicError = runtime.PrismaClientRustPanicError
  export import PrismaClientInitializationError = runtime.PrismaClientInitializationError
  export import PrismaClientValidationError = runtime.PrismaClientValidationError

  /**
   * Re-export of sql-template-tag
   */
  export import sql = runtime.sqltag
  export import empty = runtime.empty
  export import join = runtime.join
  export import raw = runtime.raw
  export import Sql = runtime.Sql



  /**
   * Decimal.js
   */
  export import Decimal = runtime.Decimal

  export type DecimalJsLike = runtime.DecimalJsLike

  /**
   * Metrics
   */
  export type Metrics = runtime.Metrics
  export type Metric<T> = runtime.Metric<T>
  export type MetricHistogram = runtime.MetricHistogram
  export type MetricHistogramBucket = runtime.MetricHistogramBucket

  /**
  * Extensions
  */
  export import Extension = $Extensions.UserArgs
  export import getExtensionContext = runtime.Extensions.getExtensionContext
  export import Args = $Public.Args
  export import Payload = $Public.Payload
  export import Result = $Public.Result
  export import Exact = $Public.Exact

  /**
   * Prisma Client JS version: 6.16.3
   * Query Engine version: bb420e667c1820a8c05a38023385f6cc7ef8e83a
   */
  export type PrismaVersion = {
    client: string
  }

  export const prismaVersion: PrismaVersion

  /**
   * Utility Types
   */


  export import JsonObject = runtime.JsonObject
  export import JsonArray = runtime.JsonArray
  export import JsonValue = runtime.JsonValue
  export import InputJsonObject = runtime.InputJsonObject
  export import InputJsonArray = runtime.InputJsonArray
  export import InputJsonValue = runtime.InputJsonValue

  /**
   * Types of the values used to represent different kinds of `null` values when working with JSON fields.
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  namespace NullTypes {
    /**
    * Type of `Prisma.DbNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.DbNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class DbNull {
      private DbNull: never
      private constructor()
    }

    /**
    * Type of `Prisma.JsonNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.JsonNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class JsonNull {
      private JsonNull: never
      private constructor()
    }

    /**
    * Type of `Prisma.AnyNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.AnyNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class AnyNull {
      private AnyNull: never
      private constructor()
    }
  }

  /**
   * Helper for filtering JSON entries that have `null` on the database (empty on the db)
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const DbNull: NullTypes.DbNull

  /**
   * Helper for filtering JSON entries that have JSON `null` values (not empty on the db)
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const JsonNull: NullTypes.JsonNull

  /**
   * Helper for filtering JSON entries that are `Prisma.DbNull` or `Prisma.JsonNull`
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const AnyNull: NullTypes.AnyNull

  type SelectAndInclude = {
    select: any
    include: any
  }

  type SelectAndOmit = {
    select: any
    omit: any
  }

  /**
   * Get the type of the value, that the Promise holds.
   */
  export type PromiseType<T extends PromiseLike<any>> = T extends PromiseLike<infer U> ? U : T;

  /**
   * Get the return type of a function which returns a Promise.
   */
  export type PromiseReturnType<T extends (...args: any) => $Utils.JsPromise<any>> = PromiseType<ReturnType<T>>

  /**
   * From T, pick a set of properties whose keys are in the union K
   */
  type Prisma__Pick<T, K extends keyof T> = {
      [P in K]: T[P];
  };


  export type Enumerable<T> = T | Array<T>;

  export type RequiredKeys<T> = {
    [K in keyof T]-?: {} extends Prisma__Pick<T, K> ? never : K
  }[keyof T]

  export type TruthyKeys<T> = keyof {
    [K in keyof T as T[K] extends false | undefined | null ? never : K]: K
  }

  export type TrueKeys<T> = TruthyKeys<Prisma__Pick<T, RequiredKeys<T>>>

  /**
   * Subset
   * @desc From `T` pick properties that exist in `U`. Simple version of Intersection
   */
  export type Subset<T, U> = {
    [key in keyof T]: key extends keyof U ? T[key] : never;
  };

  /**
   * SelectSubset
   * @desc From `T` pick properties that exist in `U`. Simple version of Intersection.
   * Additionally, it validates, if both select and include are present. If the case, it errors.
   */
  export type SelectSubset<T, U> = {
    [key in keyof T]: key extends keyof U ? T[key] : never
  } &
    (T extends SelectAndInclude
      ? 'Please either choose `select` or `include`.'
      : T extends SelectAndOmit
        ? 'Please either choose `select` or `omit`.'
        : {})

  /**
   * Subset + Intersection
   * @desc From `T` pick properties that exist in `U` and intersect `K`
   */
  export type SubsetIntersection<T, U, K> = {
    [key in keyof T]: key extends keyof U ? T[key] : never
  } &
    K

  type Without<T, U> = { [P in Exclude<keyof T, keyof U>]?: never };

  /**
   * XOR is needed to have a real mutually exclusive union type
   * https://stackoverflow.com/questions/42123407/does-typescript-support-mutually-exclusive-types
   */
  type XOR<T, U> =
    T extends object ?
    U extends object ?
      (Without<T, U> & U) | (Without<U, T> & T)
    : U : T


  /**
   * Is T a Record?
   */
  type IsObject<T extends any> = T extends Array<any>
  ? False
  : T extends Date
  ? False
  : T extends Uint8Array
  ? False
  : T extends BigInt
  ? False
  : T extends object
  ? True
  : False


  /**
   * If it's T[], return T
   */
  export type UnEnumerate<T extends unknown> = T extends Array<infer U> ? U : T

  /**
   * From ts-toolbelt
   */

  type __Either<O extends object, K extends Key> = Omit<O, K> &
    {
      // Merge all but K
      [P in K]: Prisma__Pick<O, P & keyof O> // With K possibilities
    }[K]

  type EitherStrict<O extends object, K extends Key> = Strict<__Either<O, K>>

  type EitherLoose<O extends object, K extends Key> = ComputeRaw<__Either<O, K>>

  type _Either<
    O extends object,
    K extends Key,
    strict extends Boolean
  > = {
    1: EitherStrict<O, K>
    0: EitherLoose<O, K>
  }[strict]

  type Either<
    O extends object,
    K extends Key,
    strict extends Boolean = 1
  > = O extends unknown ? _Either<O, K, strict> : never

  export type Union = any

  type PatchUndefined<O extends object, O1 extends object> = {
    [K in keyof O]: O[K] extends undefined ? At<O1, K> : O[K]
  } & {}

  /** Helper Types for "Merge" **/
  export type IntersectOf<U extends Union> = (
    U extends unknown ? (k: U) => void : never
  ) extends (k: infer I) => void
    ? I
    : never

  export type Overwrite<O extends object, O1 extends object> = {
      [K in keyof O]: K extends keyof O1 ? O1[K] : O[K];
  } & {};

  type _Merge<U extends object> = IntersectOf<Overwrite<U, {
      [K in keyof U]-?: At<U, K>;
  }>>;

  type Key = string | number | symbol;
  type AtBasic<O extends object, K extends Key> = K extends keyof O ? O[K] : never;
  type AtStrict<O extends object, K extends Key> = O[K & keyof O];
  type AtLoose<O extends object, K extends Key> = O extends unknown ? AtStrict<O, K> : never;
  export type At<O extends object, K extends Key, strict extends Boolean = 1> = {
      1: AtStrict<O, K>;
      0: AtLoose<O, K>;
  }[strict];

  export type ComputeRaw<A extends any> = A extends Function ? A : {
    [K in keyof A]: A[K];
  } & {};

  export type OptionalFlat<O> = {
    [K in keyof O]?: O[K];
  } & {};

  type _Record<K extends keyof any, T> = {
    [P in K]: T;
  };

  // cause typescript not to expand types and preserve names
  type NoExpand<T> = T extends unknown ? T : never;

  // this type assumes the passed object is entirely optional
  type AtLeast<O extends object, K extends string> = NoExpand<
    O extends unknown
    ? | (K extends keyof O ? { [P in K]: O[P] } & O : O)
      | {[P in keyof O as P extends K ? P : never]-?: O[P]} & O
    : never>;

  type _Strict<U, _U = U> = U extends unknown ? U & OptionalFlat<_Record<Exclude<Keys<_U>, keyof U>, never>> : never;

  export type Strict<U extends object> = ComputeRaw<_Strict<U>>;
  /** End Helper Types for "Merge" **/

  export type Merge<U extends object> = ComputeRaw<_Merge<Strict<U>>>;

  /**
  A [[Boolean]]
  */
  export type Boolean = True | False

  // /**
  // 1
  // */
  export type True = 1

  /**
  0
  */
  export type False = 0

  export type Not<B extends Boolean> = {
    0: 1
    1: 0
  }[B]

  export type Extends<A1 extends any, A2 extends any> = [A1] extends [never]
    ? 0 // anything `never` is false
    : A1 extends A2
    ? 1
    : 0

  export type Has<U extends Union, U1 extends Union> = Not<
    Extends<Exclude<U1, U>, U1>
  >

  export type Or<B1 extends Boolean, B2 extends Boolean> = {
    0: {
      0: 0
      1: 1
    }
    1: {
      0: 1
      1: 1
    }
  }[B1][B2]

  export type Keys<U extends Union> = U extends unknown ? keyof U : never

  type Cast<A, B> = A extends B ? A : B;

  export const type: unique symbol;



  /**
   * Used by group by
   */

  export type GetScalarType<T, O> = O extends object ? {
    [P in keyof T]: P extends keyof O
      ? O[P]
      : never
  } : never

  type FieldPaths<
    T,
    U = Omit<T, '_avg' | '_sum' | '_count' | '_min' | '_max'>
  > = IsObject<T> extends True ? U : T

  type GetHavingFields<T> = {
    [K in keyof T]: Or<
      Or<Extends<'OR', K>, Extends<'AND', K>>,
      Extends<'NOT', K>
    > extends True
      ? // infer is only needed to not hit TS limit
        // based on the brilliant idea of Pierre-Antoine Mills
        // https://github.com/microsoft/TypeScript/issues/30188#issuecomment-478938437
        T[K] extends infer TK
        ? GetHavingFields<UnEnumerate<TK> extends object ? Merge<UnEnumerate<TK>> : never>
        : never
      : {} extends FieldPaths<T[K]>
      ? never
      : K
  }[keyof T]

  /**
   * Convert tuple to union
   */
  type _TupleToUnion<T> = T extends (infer E)[] ? E : never
  type TupleToUnion<K extends readonly any[]> = _TupleToUnion<K>
  type MaybeTupleToUnion<T> = T extends any[] ? TupleToUnion<T> : T

  /**
   * Like `Pick`, but additionally can also accept an array of keys
   */
  type PickEnumerable<T, K extends Enumerable<keyof T> | keyof T> = Prisma__Pick<T, MaybeTupleToUnion<K>>

  /**
   * Exclude all keys with underscores
   */
  type ExcludeUnderscoreKeys<T extends string> = T extends `_${string}` ? never : T


  export type FieldRef<Model, FieldType> = runtime.FieldRef<Model, FieldType>

  type FieldRefInputType<Model, FieldType> = Model extends never ? never : FieldRef<Model, FieldType>


  export const ModelName: {
    Question: 'Question',
    Topic: 'Topic',
    TopicAssignment: 'TopicAssignment',
    ClusterResult: 'ClusterResult',
    AnalysisRun: 'AnalysisRun',
    SystemConfig: 'SystemConfig',
    UserSession: 'UserSession',
    DataSync: 'DataSync'
  };

  export type ModelName = (typeof ModelName)[keyof typeof ModelName]


  export type Datasources = {
    db?: Datasource
  }

  interface TypeMapCb<ClientOptions = {}> extends $Utils.Fn<{extArgs: $Extensions.InternalArgs }, $Utils.Record<string, any>> {
    returns: Prisma.TypeMap<this['params']['extArgs'], ClientOptions extends { omit: infer OmitOptions } ? OmitOptions : {}>
  }

  export type TypeMap<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> = {
    globalOmitOptions: {
      omit: GlobalOmitOptions
    }
    meta: {
      modelProps: "question" | "topic" | "topicAssignment" | "clusterResult" | "analysisRun" | "systemConfig" | "userSession" | "dataSync"
      txIsolationLevel: Prisma.TransactionIsolationLevel
    }
    model: {
      Question: {
        payload: Prisma.$QuestionPayload<ExtArgs>
        fields: Prisma.QuestionFieldRefs
        operations: {
          findUnique: {
            args: Prisma.QuestionFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.QuestionFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload>
          }
          findFirst: {
            args: Prisma.QuestionFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.QuestionFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload>
          }
          findMany: {
            args: Prisma.QuestionFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload>[]
          }
          create: {
            args: Prisma.QuestionCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload>
          }
          createMany: {
            args: Prisma.QuestionCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.QuestionCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload>[]
          }
          delete: {
            args: Prisma.QuestionDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload>
          }
          update: {
            args: Prisma.QuestionUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload>
          }
          deleteMany: {
            args: Prisma.QuestionDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.QuestionUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.QuestionUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload>[]
          }
          upsert: {
            args: Prisma.QuestionUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$QuestionPayload>
          }
          aggregate: {
            args: Prisma.QuestionAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateQuestion>
          }
          groupBy: {
            args: Prisma.QuestionGroupByArgs<ExtArgs>
            result: $Utils.Optional<QuestionGroupByOutputType>[]
          }
          count: {
            args: Prisma.QuestionCountArgs<ExtArgs>
            result: $Utils.Optional<QuestionCountAggregateOutputType> | number
          }
        }
      }
      Topic: {
        payload: Prisma.$TopicPayload<ExtArgs>
        fields: Prisma.TopicFieldRefs
        operations: {
          findUnique: {
            args: Prisma.TopicFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.TopicFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload>
          }
          findFirst: {
            args: Prisma.TopicFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.TopicFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload>
          }
          findMany: {
            args: Prisma.TopicFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload>[]
          }
          create: {
            args: Prisma.TopicCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload>
          }
          createMany: {
            args: Prisma.TopicCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.TopicCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload>[]
          }
          delete: {
            args: Prisma.TopicDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload>
          }
          update: {
            args: Prisma.TopicUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload>
          }
          deleteMany: {
            args: Prisma.TopicDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.TopicUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.TopicUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload>[]
          }
          upsert: {
            args: Prisma.TopicUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicPayload>
          }
          aggregate: {
            args: Prisma.TopicAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateTopic>
          }
          groupBy: {
            args: Prisma.TopicGroupByArgs<ExtArgs>
            result: $Utils.Optional<TopicGroupByOutputType>[]
          }
          count: {
            args: Prisma.TopicCountArgs<ExtArgs>
            result: $Utils.Optional<TopicCountAggregateOutputType> | number
          }
        }
      }
      TopicAssignment: {
        payload: Prisma.$TopicAssignmentPayload<ExtArgs>
        fields: Prisma.TopicAssignmentFieldRefs
        operations: {
          findUnique: {
            args: Prisma.TopicAssignmentFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.TopicAssignmentFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload>
          }
          findFirst: {
            args: Prisma.TopicAssignmentFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.TopicAssignmentFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload>
          }
          findMany: {
            args: Prisma.TopicAssignmentFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload>[]
          }
          create: {
            args: Prisma.TopicAssignmentCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload>
          }
          createMany: {
            args: Prisma.TopicAssignmentCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.TopicAssignmentCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload>[]
          }
          delete: {
            args: Prisma.TopicAssignmentDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload>
          }
          update: {
            args: Prisma.TopicAssignmentUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload>
          }
          deleteMany: {
            args: Prisma.TopicAssignmentDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.TopicAssignmentUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.TopicAssignmentUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload>[]
          }
          upsert: {
            args: Prisma.TopicAssignmentUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$TopicAssignmentPayload>
          }
          aggregate: {
            args: Prisma.TopicAssignmentAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateTopicAssignment>
          }
          groupBy: {
            args: Prisma.TopicAssignmentGroupByArgs<ExtArgs>
            result: $Utils.Optional<TopicAssignmentGroupByOutputType>[]
          }
          count: {
            args: Prisma.TopicAssignmentCountArgs<ExtArgs>
            result: $Utils.Optional<TopicAssignmentCountAggregateOutputType> | number
          }
        }
      }
      ClusterResult: {
        payload: Prisma.$ClusterResultPayload<ExtArgs>
        fields: Prisma.ClusterResultFieldRefs
        operations: {
          findUnique: {
            args: Prisma.ClusterResultFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.ClusterResultFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload>
          }
          findFirst: {
            args: Prisma.ClusterResultFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.ClusterResultFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload>
          }
          findMany: {
            args: Prisma.ClusterResultFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload>[]
          }
          create: {
            args: Prisma.ClusterResultCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload>
          }
          createMany: {
            args: Prisma.ClusterResultCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.ClusterResultCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload>[]
          }
          delete: {
            args: Prisma.ClusterResultDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload>
          }
          update: {
            args: Prisma.ClusterResultUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload>
          }
          deleteMany: {
            args: Prisma.ClusterResultDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.ClusterResultUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.ClusterResultUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload>[]
          }
          upsert: {
            args: Prisma.ClusterResultUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ClusterResultPayload>
          }
          aggregate: {
            args: Prisma.ClusterResultAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateClusterResult>
          }
          groupBy: {
            args: Prisma.ClusterResultGroupByArgs<ExtArgs>
            result: $Utils.Optional<ClusterResultGroupByOutputType>[]
          }
          count: {
            args: Prisma.ClusterResultCountArgs<ExtArgs>
            result: $Utils.Optional<ClusterResultCountAggregateOutputType> | number
          }
        }
      }
      AnalysisRun: {
        payload: Prisma.$AnalysisRunPayload<ExtArgs>
        fields: Prisma.AnalysisRunFieldRefs
        operations: {
          findUnique: {
            args: Prisma.AnalysisRunFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.AnalysisRunFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload>
          }
          findFirst: {
            args: Prisma.AnalysisRunFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.AnalysisRunFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload>
          }
          findMany: {
            args: Prisma.AnalysisRunFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload>[]
          }
          create: {
            args: Prisma.AnalysisRunCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload>
          }
          createMany: {
            args: Prisma.AnalysisRunCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.AnalysisRunCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload>[]
          }
          delete: {
            args: Prisma.AnalysisRunDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload>
          }
          update: {
            args: Prisma.AnalysisRunUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload>
          }
          deleteMany: {
            args: Prisma.AnalysisRunDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.AnalysisRunUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.AnalysisRunUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload>[]
          }
          upsert: {
            args: Prisma.AnalysisRunUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$AnalysisRunPayload>
          }
          aggregate: {
            args: Prisma.AnalysisRunAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateAnalysisRun>
          }
          groupBy: {
            args: Prisma.AnalysisRunGroupByArgs<ExtArgs>
            result: $Utils.Optional<AnalysisRunGroupByOutputType>[]
          }
          count: {
            args: Prisma.AnalysisRunCountArgs<ExtArgs>
            result: $Utils.Optional<AnalysisRunCountAggregateOutputType> | number
          }
        }
      }
      SystemConfig: {
        payload: Prisma.$SystemConfigPayload<ExtArgs>
        fields: Prisma.SystemConfigFieldRefs
        operations: {
          findUnique: {
            args: Prisma.SystemConfigFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.SystemConfigFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload>
          }
          findFirst: {
            args: Prisma.SystemConfigFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.SystemConfigFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload>
          }
          findMany: {
            args: Prisma.SystemConfigFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload>[]
          }
          create: {
            args: Prisma.SystemConfigCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload>
          }
          createMany: {
            args: Prisma.SystemConfigCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.SystemConfigCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload>[]
          }
          delete: {
            args: Prisma.SystemConfigDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload>
          }
          update: {
            args: Prisma.SystemConfigUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload>
          }
          deleteMany: {
            args: Prisma.SystemConfigDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.SystemConfigUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.SystemConfigUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload>[]
          }
          upsert: {
            args: Prisma.SystemConfigUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$SystemConfigPayload>
          }
          aggregate: {
            args: Prisma.SystemConfigAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateSystemConfig>
          }
          groupBy: {
            args: Prisma.SystemConfigGroupByArgs<ExtArgs>
            result: $Utils.Optional<SystemConfigGroupByOutputType>[]
          }
          count: {
            args: Prisma.SystemConfigCountArgs<ExtArgs>
            result: $Utils.Optional<SystemConfigCountAggregateOutputType> | number
          }
        }
      }
      UserSession: {
        payload: Prisma.$UserSessionPayload<ExtArgs>
        fields: Prisma.UserSessionFieldRefs
        operations: {
          findUnique: {
            args: Prisma.UserSessionFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.UserSessionFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload>
          }
          findFirst: {
            args: Prisma.UserSessionFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.UserSessionFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload>
          }
          findMany: {
            args: Prisma.UserSessionFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload>[]
          }
          create: {
            args: Prisma.UserSessionCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload>
          }
          createMany: {
            args: Prisma.UserSessionCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.UserSessionCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload>[]
          }
          delete: {
            args: Prisma.UserSessionDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload>
          }
          update: {
            args: Prisma.UserSessionUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload>
          }
          deleteMany: {
            args: Prisma.UserSessionDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.UserSessionUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.UserSessionUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload>[]
          }
          upsert: {
            args: Prisma.UserSessionUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserSessionPayload>
          }
          aggregate: {
            args: Prisma.UserSessionAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateUserSession>
          }
          groupBy: {
            args: Prisma.UserSessionGroupByArgs<ExtArgs>
            result: $Utils.Optional<UserSessionGroupByOutputType>[]
          }
          count: {
            args: Prisma.UserSessionCountArgs<ExtArgs>
            result: $Utils.Optional<UserSessionCountAggregateOutputType> | number
          }
        }
      }
      DataSync: {
        payload: Prisma.$DataSyncPayload<ExtArgs>
        fields: Prisma.DataSyncFieldRefs
        operations: {
          findUnique: {
            args: Prisma.DataSyncFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.DataSyncFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload>
          }
          findFirst: {
            args: Prisma.DataSyncFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.DataSyncFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload>
          }
          findMany: {
            args: Prisma.DataSyncFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload>[]
          }
          create: {
            args: Prisma.DataSyncCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload>
          }
          createMany: {
            args: Prisma.DataSyncCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.DataSyncCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload>[]
          }
          delete: {
            args: Prisma.DataSyncDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload>
          }
          update: {
            args: Prisma.DataSyncUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload>
          }
          deleteMany: {
            args: Prisma.DataSyncDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.DataSyncUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.DataSyncUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload>[]
          }
          upsert: {
            args: Prisma.DataSyncUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSyncPayload>
          }
          aggregate: {
            args: Prisma.DataSyncAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateDataSync>
          }
          groupBy: {
            args: Prisma.DataSyncGroupByArgs<ExtArgs>
            result: $Utils.Optional<DataSyncGroupByOutputType>[]
          }
          count: {
            args: Prisma.DataSyncCountArgs<ExtArgs>
            result: $Utils.Optional<DataSyncCountAggregateOutputType> | number
          }
        }
      }
    }
  } & {
    other: {
      payload: any
      operations: {
        $executeRaw: {
          args: [query: TemplateStringsArray | Prisma.Sql, ...values: any[]],
          result: any
        }
        $executeRawUnsafe: {
          args: [query: string, ...values: any[]],
          result: any
        }
        $queryRaw: {
          args: [query: TemplateStringsArray | Prisma.Sql, ...values: any[]],
          result: any
        }
        $queryRawUnsafe: {
          args: [query: string, ...values: any[]],
          result: any
        }
      }
    }
  }
  export const defineExtension: $Extensions.ExtendsHook<"define", Prisma.TypeMapCb, $Extensions.DefaultArgs>
  export type DefaultPrismaClient = PrismaClient
  export type ErrorFormat = 'pretty' | 'colorless' | 'minimal'
  export interface PrismaClientOptions {
    /**
     * Overwrites the datasource url from your schema.prisma file
     */
    datasources?: Datasources
    /**
     * Overwrites the datasource url from your schema.prisma file
     */
    datasourceUrl?: string
    /**
     * @default "colorless"
     */
    errorFormat?: ErrorFormat
    /**
     * @example
     * ```
     * // Shorthand for `emit: 'stdout'`
     * log: ['query', 'info', 'warn', 'error']
     * 
     * // Emit as events only
     * log: [
     *   { emit: 'event', level: 'query' },
     *   { emit: 'event', level: 'info' },
     *   { emit: 'event', level: 'warn' }
     *   { emit: 'event', level: 'error' }
     * ]
     * 
     * / Emit as events and log to stdout
     * og: [
     *  { emit: 'stdout', level: 'query' },
     *  { emit: 'stdout', level: 'info' },
     *  { emit: 'stdout', level: 'warn' }
     *  { emit: 'stdout', level: 'error' }
     * 
     * ```
     * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/logging#the-log-option).
     */
    log?: (LogLevel | LogDefinition)[]
    /**
     * The default values for transactionOptions
     * maxWait ?= 2000
     * timeout ?= 5000
     */
    transactionOptions?: {
      maxWait?: number
      timeout?: number
      isolationLevel?: Prisma.TransactionIsolationLevel
    }
    /**
     * Instance of a Driver Adapter, e.g., like one provided by `@prisma/adapter-planetscale`
     */
    adapter?: runtime.SqlDriverAdapterFactory | null
    /**
     * Global configuration for omitting model fields by default.
     * 
     * @example
     * ```
     * const prisma = new PrismaClient({
     *   omit: {
     *     user: {
     *       password: true
     *     }
     *   }
     * })
     * ```
     */
    omit?: Prisma.GlobalOmitConfig
  }
  export type GlobalOmitConfig = {
    question?: QuestionOmit
    topic?: TopicOmit
    topicAssignment?: TopicAssignmentOmit
    clusterResult?: ClusterResultOmit
    analysisRun?: AnalysisRunOmit
    systemConfig?: SystemConfigOmit
    userSession?: UserSessionOmit
    dataSync?: DataSyncOmit
  }

  /* Types for Logging */
  export type LogLevel = 'info' | 'query' | 'warn' | 'error'
  export type LogDefinition = {
    level: LogLevel
    emit: 'stdout' | 'event'
  }

  export type CheckIsLogLevel<T> = T extends LogLevel ? T : never;

  export type GetLogType<T> = CheckIsLogLevel<
    T extends LogDefinition ? T['level'] : T
  >;

  export type GetEvents<T extends any[]> = T extends Array<LogLevel | LogDefinition>
    ? GetLogType<T[number]>
    : never;

  export type QueryEvent = {
    timestamp: Date
    query: string
    params: string
    duration: number
    target: string
  }

  export type LogEvent = {
    timestamp: Date
    message: string
    target: string
  }
  /* End Types for Logging */


  export type PrismaAction =
    | 'findUnique'
    | 'findUniqueOrThrow'
    | 'findMany'
    | 'findFirst'
    | 'findFirstOrThrow'
    | 'create'
    | 'createMany'
    | 'createManyAndReturn'
    | 'update'
    | 'updateMany'
    | 'updateManyAndReturn'
    | 'upsert'
    | 'delete'
    | 'deleteMany'
    | 'executeRaw'
    | 'queryRaw'
    | 'aggregate'
    | 'count'
    | 'runCommandRaw'
    | 'findRaw'
    | 'groupBy'

  // tested in getLogLevel.test.ts
  export function getLogLevel(log: Array<LogLevel | LogDefinition>): LogLevel | undefined;

  /**
   * `PrismaClient` proxy available in interactive transactions.
   */
  export type TransactionClient = Omit<Prisma.DefaultPrismaClient, runtime.ITXClientDenyList>

  export type Datasource = {
    url?: string
  }

  /**
   * Count Types
   */


  /**
   * Count Type QuestionCountOutputType
   */

  export type QuestionCountOutputType = {
    topicAssignments: number
    clusterResults: number
  }

  export type QuestionCountOutputTypeSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    topicAssignments?: boolean | QuestionCountOutputTypeCountTopicAssignmentsArgs
    clusterResults?: boolean | QuestionCountOutputTypeCountClusterResultsArgs
  }

  // Custom InputTypes
  /**
   * QuestionCountOutputType without action
   */
  export type QuestionCountOutputTypeDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the QuestionCountOutputType
     */
    select?: QuestionCountOutputTypeSelect<ExtArgs> | null
  }

  /**
   * QuestionCountOutputType without action
   */
  export type QuestionCountOutputTypeCountTopicAssignmentsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: TopicAssignmentWhereInput
  }

  /**
   * QuestionCountOutputType without action
   */
  export type QuestionCountOutputTypeCountClusterResultsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: ClusterResultWhereInput
  }


  /**
   * Count Type TopicCountOutputType
   */

  export type TopicCountOutputType = {
    topicAssignments: number
  }

  export type TopicCountOutputTypeSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    topicAssignments?: boolean | TopicCountOutputTypeCountTopicAssignmentsArgs
  }

  // Custom InputTypes
  /**
   * TopicCountOutputType without action
   */
  export type TopicCountOutputTypeDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicCountOutputType
     */
    select?: TopicCountOutputTypeSelect<ExtArgs> | null
  }

  /**
   * TopicCountOutputType without action
   */
  export type TopicCountOutputTypeCountTopicAssignmentsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: TopicAssignmentWhereInput
  }


  /**
   * Count Type AnalysisRunCountOutputType
   */

  export type AnalysisRunCountOutputType = {
    topicAssignments: number
    clusterResults: number
  }

  export type AnalysisRunCountOutputTypeSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    topicAssignments?: boolean | AnalysisRunCountOutputTypeCountTopicAssignmentsArgs
    clusterResults?: boolean | AnalysisRunCountOutputTypeCountClusterResultsArgs
  }

  // Custom InputTypes
  /**
   * AnalysisRunCountOutputType without action
   */
  export type AnalysisRunCountOutputTypeDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRunCountOutputType
     */
    select?: AnalysisRunCountOutputTypeSelect<ExtArgs> | null
  }

  /**
   * AnalysisRunCountOutputType without action
   */
  export type AnalysisRunCountOutputTypeCountTopicAssignmentsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: TopicAssignmentWhereInput
  }

  /**
   * AnalysisRunCountOutputType without action
   */
  export type AnalysisRunCountOutputTypeCountClusterResultsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: ClusterResultWhereInput
  }


  /**
   * Models
   */

  /**
   * Model Question
   */

  export type AggregateQuestion = {
    _count: QuestionCountAggregateOutputType | null
    _avg: QuestionAvgAggregateOutputType | null
    _sum: QuestionSumAggregateOutputType | null
    _min: QuestionMinAggregateOutputType | null
    _max: QuestionMaxAggregateOutputType | null
  }

  export type QuestionAvgAggregateOutputType = {
    embedding: number | null
  }

  export type QuestionSumAggregateOutputType = {
    embedding: number[]
  }

  export type QuestionMinAggregateOutputType = {
    id: string | null
    text: string | null
    originalText: string | null
    language: string | null
    country: string | null
    state: string | null
    userId: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type QuestionMaxAggregateOutputType = {
    id: string | null
    text: string | null
    originalText: string | null
    language: string | null
    country: string | null
    state: string | null
    userId: string | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type QuestionCountAggregateOutputType = {
    id: number
    text: number
    originalText: number
    embedding: number
    language: number
    country: number
    state: number
    userId: number
    createdAt: number
    updatedAt: number
    _all: number
  }


  export type QuestionAvgAggregateInputType = {
    embedding?: true
  }

  export type QuestionSumAggregateInputType = {
    embedding?: true
  }

  export type QuestionMinAggregateInputType = {
    id?: true
    text?: true
    originalText?: true
    language?: true
    country?: true
    state?: true
    userId?: true
    createdAt?: true
    updatedAt?: true
  }

  export type QuestionMaxAggregateInputType = {
    id?: true
    text?: true
    originalText?: true
    language?: true
    country?: true
    state?: true
    userId?: true
    createdAt?: true
    updatedAt?: true
  }

  export type QuestionCountAggregateInputType = {
    id?: true
    text?: true
    originalText?: true
    embedding?: true
    language?: true
    country?: true
    state?: true
    userId?: true
    createdAt?: true
    updatedAt?: true
    _all?: true
  }

  export type QuestionAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Question to aggregate.
     */
    where?: QuestionWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Questions to fetch.
     */
    orderBy?: QuestionOrderByWithRelationInput | QuestionOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: QuestionWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Questions from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Questions.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned Questions
    **/
    _count?: true | QuestionCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: QuestionAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: QuestionSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: QuestionMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: QuestionMaxAggregateInputType
  }

  export type GetQuestionAggregateType<T extends QuestionAggregateArgs> = {
        [P in keyof T & keyof AggregateQuestion]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateQuestion[P]>
      : GetScalarType<T[P], AggregateQuestion[P]>
  }




  export type QuestionGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: QuestionWhereInput
    orderBy?: QuestionOrderByWithAggregationInput | QuestionOrderByWithAggregationInput[]
    by: QuestionScalarFieldEnum[] | QuestionScalarFieldEnum
    having?: QuestionScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: QuestionCountAggregateInputType | true
    _avg?: QuestionAvgAggregateInputType
    _sum?: QuestionSumAggregateInputType
    _min?: QuestionMinAggregateInputType
    _max?: QuestionMaxAggregateInputType
  }

  export type QuestionGroupByOutputType = {
    id: string
    text: string
    originalText: string | null
    embedding: number[]
    language: string
    country: string | null
    state: string | null
    userId: string | null
    createdAt: Date
    updatedAt: Date
    _count: QuestionCountAggregateOutputType | null
    _avg: QuestionAvgAggregateOutputType | null
    _sum: QuestionSumAggregateOutputType | null
    _min: QuestionMinAggregateOutputType | null
    _max: QuestionMaxAggregateOutputType | null
  }

  type GetQuestionGroupByPayload<T extends QuestionGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<QuestionGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof QuestionGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], QuestionGroupByOutputType[P]>
            : GetScalarType<T[P], QuestionGroupByOutputType[P]>
        }
      >
    >


  export type QuestionSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    text?: boolean
    originalText?: boolean
    embedding?: boolean
    language?: boolean
    country?: boolean
    state?: boolean
    userId?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    topicAssignments?: boolean | Question$topicAssignmentsArgs<ExtArgs>
    clusterResults?: boolean | Question$clusterResultsArgs<ExtArgs>
    _count?: boolean | QuestionCountOutputTypeDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["question"]>

  export type QuestionSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    text?: boolean
    originalText?: boolean
    embedding?: boolean
    language?: boolean
    country?: boolean
    state?: boolean
    userId?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }, ExtArgs["result"]["question"]>

  export type QuestionSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    text?: boolean
    originalText?: boolean
    embedding?: boolean
    language?: boolean
    country?: boolean
    state?: boolean
    userId?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }, ExtArgs["result"]["question"]>

  export type QuestionSelectScalar = {
    id?: boolean
    text?: boolean
    originalText?: boolean
    embedding?: boolean
    language?: boolean
    country?: boolean
    state?: boolean
    userId?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }

  export type QuestionOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "text" | "originalText" | "embedding" | "language" | "country" | "state" | "userId" | "createdAt" | "updatedAt", ExtArgs["result"]["question"]>
  export type QuestionInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    topicAssignments?: boolean | Question$topicAssignmentsArgs<ExtArgs>
    clusterResults?: boolean | Question$clusterResultsArgs<ExtArgs>
    _count?: boolean | QuestionCountOutputTypeDefaultArgs<ExtArgs>
  }
  export type QuestionIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}
  export type QuestionIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}

  export type $QuestionPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "Question"
    objects: {
      topicAssignments: Prisma.$TopicAssignmentPayload<ExtArgs>[]
      clusterResults: Prisma.$ClusterResultPayload<ExtArgs>[]
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      text: string
      originalText: string | null
      embedding: number[]
      language: string
      country: string | null
      state: string | null
      userId: string | null
      createdAt: Date
      updatedAt: Date
    }, ExtArgs["result"]["question"]>
    composites: {}
  }

  type QuestionGetPayload<S extends boolean | null | undefined | QuestionDefaultArgs> = $Result.GetResult<Prisma.$QuestionPayload, S>

  type QuestionCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<QuestionFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: QuestionCountAggregateInputType | true
    }

  export interface QuestionDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['Question'], meta: { name: 'Question' } }
    /**
     * Find zero or one Question that matches the filter.
     * @param {QuestionFindUniqueArgs} args - Arguments to find a Question
     * @example
     * // Get one Question
     * const question = await prisma.question.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends QuestionFindUniqueArgs>(args: SelectSubset<T, QuestionFindUniqueArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one Question that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {QuestionFindUniqueOrThrowArgs} args - Arguments to find a Question
     * @example
     * // Get one Question
     * const question = await prisma.question.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends QuestionFindUniqueOrThrowArgs>(args: SelectSubset<T, QuestionFindUniqueOrThrowArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first Question that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {QuestionFindFirstArgs} args - Arguments to find a Question
     * @example
     * // Get one Question
     * const question = await prisma.question.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends QuestionFindFirstArgs>(args?: SelectSubset<T, QuestionFindFirstArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first Question that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {QuestionFindFirstOrThrowArgs} args - Arguments to find a Question
     * @example
     * // Get one Question
     * const question = await prisma.question.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends QuestionFindFirstOrThrowArgs>(args?: SelectSubset<T, QuestionFindFirstOrThrowArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more Questions that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {QuestionFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all Questions
     * const questions = await prisma.question.findMany()
     * 
     * // Get first 10 Questions
     * const questions = await prisma.question.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const questionWithIdOnly = await prisma.question.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends QuestionFindManyArgs>(args?: SelectSubset<T, QuestionFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a Question.
     * @param {QuestionCreateArgs} args - Arguments to create a Question.
     * @example
     * // Create one Question
     * const Question = await prisma.question.create({
     *   data: {
     *     // ... data to create a Question
     *   }
     * })
     * 
     */
    create<T extends QuestionCreateArgs>(args: SelectSubset<T, QuestionCreateArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many Questions.
     * @param {QuestionCreateManyArgs} args - Arguments to create many Questions.
     * @example
     * // Create many Questions
     * const question = await prisma.question.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends QuestionCreateManyArgs>(args?: SelectSubset<T, QuestionCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many Questions and returns the data saved in the database.
     * @param {QuestionCreateManyAndReturnArgs} args - Arguments to create many Questions.
     * @example
     * // Create many Questions
     * const question = await prisma.question.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many Questions and only return the `id`
     * const questionWithIdOnly = await prisma.question.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends QuestionCreateManyAndReturnArgs>(args?: SelectSubset<T, QuestionCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a Question.
     * @param {QuestionDeleteArgs} args - Arguments to delete one Question.
     * @example
     * // Delete one Question
     * const Question = await prisma.question.delete({
     *   where: {
     *     // ... filter to delete one Question
     *   }
     * })
     * 
     */
    delete<T extends QuestionDeleteArgs>(args: SelectSubset<T, QuestionDeleteArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one Question.
     * @param {QuestionUpdateArgs} args - Arguments to update one Question.
     * @example
     * // Update one Question
     * const question = await prisma.question.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends QuestionUpdateArgs>(args: SelectSubset<T, QuestionUpdateArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more Questions.
     * @param {QuestionDeleteManyArgs} args - Arguments to filter Questions to delete.
     * @example
     * // Delete a few Questions
     * const { count } = await prisma.question.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends QuestionDeleteManyArgs>(args?: SelectSubset<T, QuestionDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Questions.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {QuestionUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many Questions
     * const question = await prisma.question.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends QuestionUpdateManyArgs>(args: SelectSubset<T, QuestionUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Questions and returns the data updated in the database.
     * @param {QuestionUpdateManyAndReturnArgs} args - Arguments to update many Questions.
     * @example
     * // Update many Questions
     * const question = await prisma.question.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more Questions and only return the `id`
     * const questionWithIdOnly = await prisma.question.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends QuestionUpdateManyAndReturnArgs>(args: SelectSubset<T, QuestionUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one Question.
     * @param {QuestionUpsertArgs} args - Arguments to update or create a Question.
     * @example
     * // Update or create a Question
     * const question = await prisma.question.upsert({
     *   create: {
     *     // ... data to create a Question
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the Question we want to update
     *   }
     * })
     */
    upsert<T extends QuestionUpsertArgs>(args: SelectSubset<T, QuestionUpsertArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of Questions.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {QuestionCountArgs} args - Arguments to filter Questions to count.
     * @example
     * // Count the number of Questions
     * const count = await prisma.question.count({
     *   where: {
     *     // ... the filter for the Questions we want to count
     *   }
     * })
    **/
    count<T extends QuestionCountArgs>(
      args?: Subset<T, QuestionCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], QuestionCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a Question.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {QuestionAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends QuestionAggregateArgs>(args: Subset<T, QuestionAggregateArgs>): Prisma.PrismaPromise<GetQuestionAggregateType<T>>

    /**
     * Group by Question.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {QuestionGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends QuestionGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: QuestionGroupByArgs['orderBy'] }
        : { orderBy?: QuestionGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, QuestionGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetQuestionGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the Question model
   */
  readonly fields: QuestionFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for Question.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__QuestionClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    topicAssignments<T extends Question$topicAssignmentsArgs<ExtArgs> = {}>(args?: Subset<T, Question$topicAssignmentsArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    clusterResults<T extends Question$clusterResultsArgs<ExtArgs> = {}>(args?: Subset<T, Question$clusterResultsArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the Question model
   */
  interface QuestionFieldRefs {
    readonly id: FieldRef<"Question", 'String'>
    readonly text: FieldRef<"Question", 'String'>
    readonly originalText: FieldRef<"Question", 'String'>
    readonly embedding: FieldRef<"Question", 'Float[]'>
    readonly language: FieldRef<"Question", 'String'>
    readonly country: FieldRef<"Question", 'String'>
    readonly state: FieldRef<"Question", 'String'>
    readonly userId: FieldRef<"Question", 'String'>
    readonly createdAt: FieldRef<"Question", 'DateTime'>
    readonly updatedAt: FieldRef<"Question", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * Question findUnique
   */
  export type QuestionFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
    /**
     * Filter, which Question to fetch.
     */
    where: QuestionWhereUniqueInput
  }

  /**
   * Question findUniqueOrThrow
   */
  export type QuestionFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
    /**
     * Filter, which Question to fetch.
     */
    where: QuestionWhereUniqueInput
  }

  /**
   * Question findFirst
   */
  export type QuestionFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
    /**
     * Filter, which Question to fetch.
     */
    where?: QuestionWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Questions to fetch.
     */
    orderBy?: QuestionOrderByWithRelationInput | QuestionOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Questions.
     */
    cursor?: QuestionWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Questions from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Questions.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Questions.
     */
    distinct?: QuestionScalarFieldEnum | QuestionScalarFieldEnum[]
  }

  /**
   * Question findFirstOrThrow
   */
  export type QuestionFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
    /**
     * Filter, which Question to fetch.
     */
    where?: QuestionWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Questions to fetch.
     */
    orderBy?: QuestionOrderByWithRelationInput | QuestionOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Questions.
     */
    cursor?: QuestionWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Questions from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Questions.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Questions.
     */
    distinct?: QuestionScalarFieldEnum | QuestionScalarFieldEnum[]
  }

  /**
   * Question findMany
   */
  export type QuestionFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
    /**
     * Filter, which Questions to fetch.
     */
    where?: QuestionWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Questions to fetch.
     */
    orderBy?: QuestionOrderByWithRelationInput | QuestionOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing Questions.
     */
    cursor?: QuestionWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Questions from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Questions.
     */
    skip?: number
    distinct?: QuestionScalarFieldEnum | QuestionScalarFieldEnum[]
  }

  /**
   * Question create
   */
  export type QuestionCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
    /**
     * The data needed to create a Question.
     */
    data: XOR<QuestionCreateInput, QuestionUncheckedCreateInput>
  }

  /**
   * Question createMany
   */
  export type QuestionCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many Questions.
     */
    data: QuestionCreateManyInput | QuestionCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * Question createManyAndReturn
   */
  export type QuestionCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * The data used to create many Questions.
     */
    data: QuestionCreateManyInput | QuestionCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * Question update
   */
  export type QuestionUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
    /**
     * The data needed to update a Question.
     */
    data: XOR<QuestionUpdateInput, QuestionUncheckedUpdateInput>
    /**
     * Choose, which Question to update.
     */
    where: QuestionWhereUniqueInput
  }

  /**
   * Question updateMany
   */
  export type QuestionUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update Questions.
     */
    data: XOR<QuestionUpdateManyMutationInput, QuestionUncheckedUpdateManyInput>
    /**
     * Filter which Questions to update
     */
    where?: QuestionWhereInput
    /**
     * Limit how many Questions to update.
     */
    limit?: number
  }

  /**
   * Question updateManyAndReturn
   */
  export type QuestionUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * The data used to update Questions.
     */
    data: XOR<QuestionUpdateManyMutationInput, QuestionUncheckedUpdateManyInput>
    /**
     * Filter which Questions to update
     */
    where?: QuestionWhereInput
    /**
     * Limit how many Questions to update.
     */
    limit?: number
  }

  /**
   * Question upsert
   */
  export type QuestionUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
    /**
     * The filter to search for the Question to update in case it exists.
     */
    where: QuestionWhereUniqueInput
    /**
     * In case the Question found by the `where` argument doesn't exist, create a new Question with this data.
     */
    create: XOR<QuestionCreateInput, QuestionUncheckedCreateInput>
    /**
     * In case the Question was found with the provided `where` argument, update it with this data.
     */
    update: XOR<QuestionUpdateInput, QuestionUncheckedUpdateInput>
  }

  /**
   * Question delete
   */
  export type QuestionDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
    /**
     * Filter which Question to delete.
     */
    where: QuestionWhereUniqueInput
  }

  /**
   * Question deleteMany
   */
  export type QuestionDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Questions to delete
     */
    where?: QuestionWhereInput
    /**
     * Limit how many Questions to delete.
     */
    limit?: number
  }

  /**
   * Question.topicAssignments
   */
  export type Question$topicAssignmentsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    where?: TopicAssignmentWhereInput
    orderBy?: TopicAssignmentOrderByWithRelationInput | TopicAssignmentOrderByWithRelationInput[]
    cursor?: TopicAssignmentWhereUniqueInput
    take?: number
    skip?: number
    distinct?: TopicAssignmentScalarFieldEnum | TopicAssignmentScalarFieldEnum[]
  }

  /**
   * Question.clusterResults
   */
  export type Question$clusterResultsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    where?: ClusterResultWhereInput
    orderBy?: ClusterResultOrderByWithRelationInput | ClusterResultOrderByWithRelationInput[]
    cursor?: ClusterResultWhereUniqueInput
    take?: number
    skip?: number
    distinct?: ClusterResultScalarFieldEnum | ClusterResultScalarFieldEnum[]
  }

  /**
   * Question without action
   */
  export type QuestionDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Question
     */
    select?: QuestionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Question
     */
    omit?: QuestionOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: QuestionInclude<ExtArgs> | null
  }


  /**
   * Model Topic
   */

  export type AggregateTopic = {
    _count: TopicCountAggregateOutputType | null
    _avg: TopicAvgAggregateOutputType | null
    _sum: TopicSumAggregateOutputType | null
    _min: TopicMinAggregateOutputType | null
    _max: TopicMaxAggregateOutputType | null
  }

  export type TopicAvgAggregateOutputType = {
    embedding: number | null
  }

  export type TopicSumAggregateOutputType = {
    embedding: number[]
  }

  export type TopicMinAggregateOutputType = {
    id: string | null
    name: string | null
    description: string | null
    subtopic: string | null
    isSystemTopic: boolean | null
    isActive: boolean | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type TopicMaxAggregateOutputType = {
    id: string | null
    name: string | null
    description: string | null
    subtopic: string | null
    isSystemTopic: boolean | null
    isActive: boolean | null
    createdAt: Date | null
    updatedAt: Date | null
  }

  export type TopicCountAggregateOutputType = {
    id: number
    name: number
    description: number
    subtopic: number
    isSystemTopic: number
    isActive: number
    embedding: number
    createdAt: number
    updatedAt: number
    _all: number
  }


  export type TopicAvgAggregateInputType = {
    embedding?: true
  }

  export type TopicSumAggregateInputType = {
    embedding?: true
  }

  export type TopicMinAggregateInputType = {
    id?: true
    name?: true
    description?: true
    subtopic?: true
    isSystemTopic?: true
    isActive?: true
    createdAt?: true
    updatedAt?: true
  }

  export type TopicMaxAggregateInputType = {
    id?: true
    name?: true
    description?: true
    subtopic?: true
    isSystemTopic?: true
    isActive?: true
    createdAt?: true
    updatedAt?: true
  }

  export type TopicCountAggregateInputType = {
    id?: true
    name?: true
    description?: true
    subtopic?: true
    isSystemTopic?: true
    isActive?: true
    embedding?: true
    createdAt?: true
    updatedAt?: true
    _all?: true
  }

  export type TopicAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Topic to aggregate.
     */
    where?: TopicWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Topics to fetch.
     */
    orderBy?: TopicOrderByWithRelationInput | TopicOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: TopicWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Topics from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Topics.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned Topics
    **/
    _count?: true | TopicCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: TopicAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: TopicSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: TopicMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: TopicMaxAggregateInputType
  }

  export type GetTopicAggregateType<T extends TopicAggregateArgs> = {
        [P in keyof T & keyof AggregateTopic]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateTopic[P]>
      : GetScalarType<T[P], AggregateTopic[P]>
  }




  export type TopicGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: TopicWhereInput
    orderBy?: TopicOrderByWithAggregationInput | TopicOrderByWithAggregationInput[]
    by: TopicScalarFieldEnum[] | TopicScalarFieldEnum
    having?: TopicScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: TopicCountAggregateInputType | true
    _avg?: TopicAvgAggregateInputType
    _sum?: TopicSumAggregateInputType
    _min?: TopicMinAggregateInputType
    _max?: TopicMaxAggregateInputType
  }

  export type TopicGroupByOutputType = {
    id: string
    name: string
    description: string | null
    subtopic: string | null
    isSystemTopic: boolean
    isActive: boolean
    embedding: number[]
    createdAt: Date
    updatedAt: Date
    _count: TopicCountAggregateOutputType | null
    _avg: TopicAvgAggregateOutputType | null
    _sum: TopicSumAggregateOutputType | null
    _min: TopicMinAggregateOutputType | null
    _max: TopicMaxAggregateOutputType | null
  }

  type GetTopicGroupByPayload<T extends TopicGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<TopicGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof TopicGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], TopicGroupByOutputType[P]>
            : GetScalarType<T[P], TopicGroupByOutputType[P]>
        }
      >
    >


  export type TopicSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    name?: boolean
    description?: boolean
    subtopic?: boolean
    isSystemTopic?: boolean
    isActive?: boolean
    embedding?: boolean
    createdAt?: boolean
    updatedAt?: boolean
    topicAssignments?: boolean | Topic$topicAssignmentsArgs<ExtArgs>
    _count?: boolean | TopicCountOutputTypeDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["topic"]>

  export type TopicSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    name?: boolean
    description?: boolean
    subtopic?: boolean
    isSystemTopic?: boolean
    isActive?: boolean
    embedding?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }, ExtArgs["result"]["topic"]>

  export type TopicSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    name?: boolean
    description?: boolean
    subtopic?: boolean
    isSystemTopic?: boolean
    isActive?: boolean
    embedding?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }, ExtArgs["result"]["topic"]>

  export type TopicSelectScalar = {
    id?: boolean
    name?: boolean
    description?: boolean
    subtopic?: boolean
    isSystemTopic?: boolean
    isActive?: boolean
    embedding?: boolean
    createdAt?: boolean
    updatedAt?: boolean
  }

  export type TopicOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "name" | "description" | "subtopic" | "isSystemTopic" | "isActive" | "embedding" | "createdAt" | "updatedAt", ExtArgs["result"]["topic"]>
  export type TopicInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    topicAssignments?: boolean | Topic$topicAssignmentsArgs<ExtArgs>
    _count?: boolean | TopicCountOutputTypeDefaultArgs<ExtArgs>
  }
  export type TopicIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}
  export type TopicIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}

  export type $TopicPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "Topic"
    objects: {
      topicAssignments: Prisma.$TopicAssignmentPayload<ExtArgs>[]
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      name: string
      description: string | null
      subtopic: string | null
      isSystemTopic: boolean
      isActive: boolean
      embedding: number[]
      createdAt: Date
      updatedAt: Date
    }, ExtArgs["result"]["topic"]>
    composites: {}
  }

  type TopicGetPayload<S extends boolean | null | undefined | TopicDefaultArgs> = $Result.GetResult<Prisma.$TopicPayload, S>

  type TopicCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<TopicFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: TopicCountAggregateInputType | true
    }

  export interface TopicDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['Topic'], meta: { name: 'Topic' } }
    /**
     * Find zero or one Topic that matches the filter.
     * @param {TopicFindUniqueArgs} args - Arguments to find a Topic
     * @example
     * // Get one Topic
     * const topic = await prisma.topic.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends TopicFindUniqueArgs>(args: SelectSubset<T, TopicFindUniqueArgs<ExtArgs>>): Prisma__TopicClient<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one Topic that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {TopicFindUniqueOrThrowArgs} args - Arguments to find a Topic
     * @example
     * // Get one Topic
     * const topic = await prisma.topic.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends TopicFindUniqueOrThrowArgs>(args: SelectSubset<T, TopicFindUniqueOrThrowArgs<ExtArgs>>): Prisma__TopicClient<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first Topic that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicFindFirstArgs} args - Arguments to find a Topic
     * @example
     * // Get one Topic
     * const topic = await prisma.topic.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends TopicFindFirstArgs>(args?: SelectSubset<T, TopicFindFirstArgs<ExtArgs>>): Prisma__TopicClient<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first Topic that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicFindFirstOrThrowArgs} args - Arguments to find a Topic
     * @example
     * // Get one Topic
     * const topic = await prisma.topic.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends TopicFindFirstOrThrowArgs>(args?: SelectSubset<T, TopicFindFirstOrThrowArgs<ExtArgs>>): Prisma__TopicClient<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more Topics that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all Topics
     * const topics = await prisma.topic.findMany()
     * 
     * // Get first 10 Topics
     * const topics = await prisma.topic.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const topicWithIdOnly = await prisma.topic.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends TopicFindManyArgs>(args?: SelectSubset<T, TopicFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a Topic.
     * @param {TopicCreateArgs} args - Arguments to create a Topic.
     * @example
     * // Create one Topic
     * const Topic = await prisma.topic.create({
     *   data: {
     *     // ... data to create a Topic
     *   }
     * })
     * 
     */
    create<T extends TopicCreateArgs>(args: SelectSubset<T, TopicCreateArgs<ExtArgs>>): Prisma__TopicClient<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many Topics.
     * @param {TopicCreateManyArgs} args - Arguments to create many Topics.
     * @example
     * // Create many Topics
     * const topic = await prisma.topic.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends TopicCreateManyArgs>(args?: SelectSubset<T, TopicCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many Topics and returns the data saved in the database.
     * @param {TopicCreateManyAndReturnArgs} args - Arguments to create many Topics.
     * @example
     * // Create many Topics
     * const topic = await prisma.topic.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many Topics and only return the `id`
     * const topicWithIdOnly = await prisma.topic.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends TopicCreateManyAndReturnArgs>(args?: SelectSubset<T, TopicCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a Topic.
     * @param {TopicDeleteArgs} args - Arguments to delete one Topic.
     * @example
     * // Delete one Topic
     * const Topic = await prisma.topic.delete({
     *   where: {
     *     // ... filter to delete one Topic
     *   }
     * })
     * 
     */
    delete<T extends TopicDeleteArgs>(args: SelectSubset<T, TopicDeleteArgs<ExtArgs>>): Prisma__TopicClient<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one Topic.
     * @param {TopicUpdateArgs} args - Arguments to update one Topic.
     * @example
     * // Update one Topic
     * const topic = await prisma.topic.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends TopicUpdateArgs>(args: SelectSubset<T, TopicUpdateArgs<ExtArgs>>): Prisma__TopicClient<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more Topics.
     * @param {TopicDeleteManyArgs} args - Arguments to filter Topics to delete.
     * @example
     * // Delete a few Topics
     * const { count } = await prisma.topic.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends TopicDeleteManyArgs>(args?: SelectSubset<T, TopicDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Topics.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many Topics
     * const topic = await prisma.topic.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends TopicUpdateManyArgs>(args: SelectSubset<T, TopicUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Topics and returns the data updated in the database.
     * @param {TopicUpdateManyAndReturnArgs} args - Arguments to update many Topics.
     * @example
     * // Update many Topics
     * const topic = await prisma.topic.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more Topics and only return the `id`
     * const topicWithIdOnly = await prisma.topic.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends TopicUpdateManyAndReturnArgs>(args: SelectSubset<T, TopicUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one Topic.
     * @param {TopicUpsertArgs} args - Arguments to update or create a Topic.
     * @example
     * // Update or create a Topic
     * const topic = await prisma.topic.upsert({
     *   create: {
     *     // ... data to create a Topic
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the Topic we want to update
     *   }
     * })
     */
    upsert<T extends TopicUpsertArgs>(args: SelectSubset<T, TopicUpsertArgs<ExtArgs>>): Prisma__TopicClient<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of Topics.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicCountArgs} args - Arguments to filter Topics to count.
     * @example
     * // Count the number of Topics
     * const count = await prisma.topic.count({
     *   where: {
     *     // ... the filter for the Topics we want to count
     *   }
     * })
    **/
    count<T extends TopicCountArgs>(
      args?: Subset<T, TopicCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], TopicCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a Topic.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends TopicAggregateArgs>(args: Subset<T, TopicAggregateArgs>): Prisma.PrismaPromise<GetTopicAggregateType<T>>

    /**
     * Group by Topic.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends TopicGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: TopicGroupByArgs['orderBy'] }
        : { orderBy?: TopicGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, TopicGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetTopicGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the Topic model
   */
  readonly fields: TopicFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for Topic.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__TopicClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    topicAssignments<T extends Topic$topicAssignmentsArgs<ExtArgs> = {}>(args?: Subset<T, Topic$topicAssignmentsArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the Topic model
   */
  interface TopicFieldRefs {
    readonly id: FieldRef<"Topic", 'String'>
    readonly name: FieldRef<"Topic", 'String'>
    readonly description: FieldRef<"Topic", 'String'>
    readonly subtopic: FieldRef<"Topic", 'String'>
    readonly isSystemTopic: FieldRef<"Topic", 'Boolean'>
    readonly isActive: FieldRef<"Topic", 'Boolean'>
    readonly embedding: FieldRef<"Topic", 'Float[]'>
    readonly createdAt: FieldRef<"Topic", 'DateTime'>
    readonly updatedAt: FieldRef<"Topic", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * Topic findUnique
   */
  export type TopicFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
    /**
     * Filter, which Topic to fetch.
     */
    where: TopicWhereUniqueInput
  }

  /**
   * Topic findUniqueOrThrow
   */
  export type TopicFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
    /**
     * Filter, which Topic to fetch.
     */
    where: TopicWhereUniqueInput
  }

  /**
   * Topic findFirst
   */
  export type TopicFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
    /**
     * Filter, which Topic to fetch.
     */
    where?: TopicWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Topics to fetch.
     */
    orderBy?: TopicOrderByWithRelationInput | TopicOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Topics.
     */
    cursor?: TopicWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Topics from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Topics.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Topics.
     */
    distinct?: TopicScalarFieldEnum | TopicScalarFieldEnum[]
  }

  /**
   * Topic findFirstOrThrow
   */
  export type TopicFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
    /**
     * Filter, which Topic to fetch.
     */
    where?: TopicWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Topics to fetch.
     */
    orderBy?: TopicOrderByWithRelationInput | TopicOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Topics.
     */
    cursor?: TopicWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Topics from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Topics.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Topics.
     */
    distinct?: TopicScalarFieldEnum | TopicScalarFieldEnum[]
  }

  /**
   * Topic findMany
   */
  export type TopicFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
    /**
     * Filter, which Topics to fetch.
     */
    where?: TopicWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Topics to fetch.
     */
    orderBy?: TopicOrderByWithRelationInput | TopicOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing Topics.
     */
    cursor?: TopicWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Topics from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Topics.
     */
    skip?: number
    distinct?: TopicScalarFieldEnum | TopicScalarFieldEnum[]
  }

  /**
   * Topic create
   */
  export type TopicCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
    /**
     * The data needed to create a Topic.
     */
    data: XOR<TopicCreateInput, TopicUncheckedCreateInput>
  }

  /**
   * Topic createMany
   */
  export type TopicCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many Topics.
     */
    data: TopicCreateManyInput | TopicCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * Topic createManyAndReturn
   */
  export type TopicCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * The data used to create many Topics.
     */
    data: TopicCreateManyInput | TopicCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * Topic update
   */
  export type TopicUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
    /**
     * The data needed to update a Topic.
     */
    data: XOR<TopicUpdateInput, TopicUncheckedUpdateInput>
    /**
     * Choose, which Topic to update.
     */
    where: TopicWhereUniqueInput
  }

  /**
   * Topic updateMany
   */
  export type TopicUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update Topics.
     */
    data: XOR<TopicUpdateManyMutationInput, TopicUncheckedUpdateManyInput>
    /**
     * Filter which Topics to update
     */
    where?: TopicWhereInput
    /**
     * Limit how many Topics to update.
     */
    limit?: number
  }

  /**
   * Topic updateManyAndReturn
   */
  export type TopicUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * The data used to update Topics.
     */
    data: XOR<TopicUpdateManyMutationInput, TopicUncheckedUpdateManyInput>
    /**
     * Filter which Topics to update
     */
    where?: TopicWhereInput
    /**
     * Limit how many Topics to update.
     */
    limit?: number
  }

  /**
   * Topic upsert
   */
  export type TopicUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
    /**
     * The filter to search for the Topic to update in case it exists.
     */
    where: TopicWhereUniqueInput
    /**
     * In case the Topic found by the `where` argument doesn't exist, create a new Topic with this data.
     */
    create: XOR<TopicCreateInput, TopicUncheckedCreateInput>
    /**
     * In case the Topic was found with the provided `where` argument, update it with this data.
     */
    update: XOR<TopicUpdateInput, TopicUncheckedUpdateInput>
  }

  /**
   * Topic delete
   */
  export type TopicDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
    /**
     * Filter which Topic to delete.
     */
    where: TopicWhereUniqueInput
  }

  /**
   * Topic deleteMany
   */
  export type TopicDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Topics to delete
     */
    where?: TopicWhereInput
    /**
     * Limit how many Topics to delete.
     */
    limit?: number
  }

  /**
   * Topic.topicAssignments
   */
  export type Topic$topicAssignmentsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    where?: TopicAssignmentWhereInput
    orderBy?: TopicAssignmentOrderByWithRelationInput | TopicAssignmentOrderByWithRelationInput[]
    cursor?: TopicAssignmentWhereUniqueInput
    take?: number
    skip?: number
    distinct?: TopicAssignmentScalarFieldEnum | TopicAssignmentScalarFieldEnum[]
  }

  /**
   * Topic without action
   */
  export type TopicDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the Topic
     */
    select?: TopicSelect<ExtArgs> | null
    /**
     * Omit specific fields from the Topic
     */
    omit?: TopicOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicInclude<ExtArgs> | null
  }


  /**
   * Model TopicAssignment
   */

  export type AggregateTopicAssignment = {
    _count: TopicAssignmentCountAggregateOutputType | null
    _avg: TopicAssignmentAvgAggregateOutputType | null
    _sum: TopicAssignmentSumAggregateOutputType | null
    _min: TopicAssignmentMinAggregateOutputType | null
    _max: TopicAssignmentMaxAggregateOutputType | null
  }

  export type TopicAssignmentAvgAggregateOutputType = {
    similarityScore: number | null
    confidence: number | null
  }

  export type TopicAssignmentSumAggregateOutputType = {
    similarityScore: number | null
    confidence: number | null
  }

  export type TopicAssignmentMinAggregateOutputType = {
    id: string | null
    questionId: string | null
    topicId: string | null
    similarityScore: number | null
    assignmentType: $Enums.AssignmentType | null
    confidence: number | null
    analysisRunId: string | null
    createdAt: Date | null
  }

  export type TopicAssignmentMaxAggregateOutputType = {
    id: string | null
    questionId: string | null
    topicId: string | null
    similarityScore: number | null
    assignmentType: $Enums.AssignmentType | null
    confidence: number | null
    analysisRunId: string | null
    createdAt: Date | null
  }

  export type TopicAssignmentCountAggregateOutputType = {
    id: number
    questionId: number
    topicId: number
    similarityScore: number
    assignmentType: number
    confidence: number
    analysisRunId: number
    createdAt: number
    _all: number
  }


  export type TopicAssignmentAvgAggregateInputType = {
    similarityScore?: true
    confidence?: true
  }

  export type TopicAssignmentSumAggregateInputType = {
    similarityScore?: true
    confidence?: true
  }

  export type TopicAssignmentMinAggregateInputType = {
    id?: true
    questionId?: true
    topicId?: true
    similarityScore?: true
    assignmentType?: true
    confidence?: true
    analysisRunId?: true
    createdAt?: true
  }

  export type TopicAssignmentMaxAggregateInputType = {
    id?: true
    questionId?: true
    topicId?: true
    similarityScore?: true
    assignmentType?: true
    confidence?: true
    analysisRunId?: true
    createdAt?: true
  }

  export type TopicAssignmentCountAggregateInputType = {
    id?: true
    questionId?: true
    topicId?: true
    similarityScore?: true
    assignmentType?: true
    confidence?: true
    analysisRunId?: true
    createdAt?: true
    _all?: true
  }

  export type TopicAssignmentAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which TopicAssignment to aggregate.
     */
    where?: TopicAssignmentWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of TopicAssignments to fetch.
     */
    orderBy?: TopicAssignmentOrderByWithRelationInput | TopicAssignmentOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: TopicAssignmentWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` TopicAssignments from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` TopicAssignments.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned TopicAssignments
    **/
    _count?: true | TopicAssignmentCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: TopicAssignmentAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: TopicAssignmentSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: TopicAssignmentMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: TopicAssignmentMaxAggregateInputType
  }

  export type GetTopicAssignmentAggregateType<T extends TopicAssignmentAggregateArgs> = {
        [P in keyof T & keyof AggregateTopicAssignment]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateTopicAssignment[P]>
      : GetScalarType<T[P], AggregateTopicAssignment[P]>
  }




  export type TopicAssignmentGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: TopicAssignmentWhereInput
    orderBy?: TopicAssignmentOrderByWithAggregationInput | TopicAssignmentOrderByWithAggregationInput[]
    by: TopicAssignmentScalarFieldEnum[] | TopicAssignmentScalarFieldEnum
    having?: TopicAssignmentScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: TopicAssignmentCountAggregateInputType | true
    _avg?: TopicAssignmentAvgAggregateInputType
    _sum?: TopicAssignmentSumAggregateInputType
    _min?: TopicAssignmentMinAggregateInputType
    _max?: TopicAssignmentMaxAggregateInputType
  }

  export type TopicAssignmentGroupByOutputType = {
    id: string
    questionId: string
    topicId: string
    similarityScore: number | null
    assignmentType: $Enums.AssignmentType
    confidence: number | null
    analysisRunId: string
    createdAt: Date
    _count: TopicAssignmentCountAggregateOutputType | null
    _avg: TopicAssignmentAvgAggregateOutputType | null
    _sum: TopicAssignmentSumAggregateOutputType | null
    _min: TopicAssignmentMinAggregateOutputType | null
    _max: TopicAssignmentMaxAggregateOutputType | null
  }

  type GetTopicAssignmentGroupByPayload<T extends TopicAssignmentGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<TopicAssignmentGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof TopicAssignmentGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], TopicAssignmentGroupByOutputType[P]>
            : GetScalarType<T[P], TopicAssignmentGroupByOutputType[P]>
        }
      >
    >


  export type TopicAssignmentSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    questionId?: boolean
    topicId?: boolean
    similarityScore?: boolean
    assignmentType?: boolean
    confidence?: boolean
    analysisRunId?: boolean
    createdAt?: boolean
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    topic?: boolean | TopicDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["topicAssignment"]>

  export type TopicAssignmentSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    questionId?: boolean
    topicId?: boolean
    similarityScore?: boolean
    assignmentType?: boolean
    confidence?: boolean
    analysisRunId?: boolean
    createdAt?: boolean
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    topic?: boolean | TopicDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["topicAssignment"]>

  export type TopicAssignmentSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    questionId?: boolean
    topicId?: boolean
    similarityScore?: boolean
    assignmentType?: boolean
    confidence?: boolean
    analysisRunId?: boolean
    createdAt?: boolean
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    topic?: boolean | TopicDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["topicAssignment"]>

  export type TopicAssignmentSelectScalar = {
    id?: boolean
    questionId?: boolean
    topicId?: boolean
    similarityScore?: boolean
    assignmentType?: boolean
    confidence?: boolean
    analysisRunId?: boolean
    createdAt?: boolean
  }

  export type TopicAssignmentOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "questionId" | "topicId" | "similarityScore" | "assignmentType" | "confidence" | "analysisRunId" | "createdAt", ExtArgs["result"]["topicAssignment"]>
  export type TopicAssignmentInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    topic?: boolean | TopicDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }
  export type TopicAssignmentIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    topic?: boolean | TopicDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }
  export type TopicAssignmentIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    topic?: boolean | TopicDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }

  export type $TopicAssignmentPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "TopicAssignment"
    objects: {
      question: Prisma.$QuestionPayload<ExtArgs>
      topic: Prisma.$TopicPayload<ExtArgs>
      analysisRun: Prisma.$AnalysisRunPayload<ExtArgs>
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      questionId: string
      topicId: string
      similarityScore: number | null
      assignmentType: $Enums.AssignmentType
      confidence: number | null
      analysisRunId: string
      createdAt: Date
    }, ExtArgs["result"]["topicAssignment"]>
    composites: {}
  }

  type TopicAssignmentGetPayload<S extends boolean | null | undefined | TopicAssignmentDefaultArgs> = $Result.GetResult<Prisma.$TopicAssignmentPayload, S>

  type TopicAssignmentCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<TopicAssignmentFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: TopicAssignmentCountAggregateInputType | true
    }

  export interface TopicAssignmentDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['TopicAssignment'], meta: { name: 'TopicAssignment' } }
    /**
     * Find zero or one TopicAssignment that matches the filter.
     * @param {TopicAssignmentFindUniqueArgs} args - Arguments to find a TopicAssignment
     * @example
     * // Get one TopicAssignment
     * const topicAssignment = await prisma.topicAssignment.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends TopicAssignmentFindUniqueArgs>(args: SelectSubset<T, TopicAssignmentFindUniqueArgs<ExtArgs>>): Prisma__TopicAssignmentClient<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one TopicAssignment that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {TopicAssignmentFindUniqueOrThrowArgs} args - Arguments to find a TopicAssignment
     * @example
     * // Get one TopicAssignment
     * const topicAssignment = await prisma.topicAssignment.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends TopicAssignmentFindUniqueOrThrowArgs>(args: SelectSubset<T, TopicAssignmentFindUniqueOrThrowArgs<ExtArgs>>): Prisma__TopicAssignmentClient<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first TopicAssignment that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicAssignmentFindFirstArgs} args - Arguments to find a TopicAssignment
     * @example
     * // Get one TopicAssignment
     * const topicAssignment = await prisma.topicAssignment.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends TopicAssignmentFindFirstArgs>(args?: SelectSubset<T, TopicAssignmentFindFirstArgs<ExtArgs>>): Prisma__TopicAssignmentClient<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first TopicAssignment that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicAssignmentFindFirstOrThrowArgs} args - Arguments to find a TopicAssignment
     * @example
     * // Get one TopicAssignment
     * const topicAssignment = await prisma.topicAssignment.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends TopicAssignmentFindFirstOrThrowArgs>(args?: SelectSubset<T, TopicAssignmentFindFirstOrThrowArgs<ExtArgs>>): Prisma__TopicAssignmentClient<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more TopicAssignments that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicAssignmentFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all TopicAssignments
     * const topicAssignments = await prisma.topicAssignment.findMany()
     * 
     * // Get first 10 TopicAssignments
     * const topicAssignments = await prisma.topicAssignment.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const topicAssignmentWithIdOnly = await prisma.topicAssignment.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends TopicAssignmentFindManyArgs>(args?: SelectSubset<T, TopicAssignmentFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a TopicAssignment.
     * @param {TopicAssignmentCreateArgs} args - Arguments to create a TopicAssignment.
     * @example
     * // Create one TopicAssignment
     * const TopicAssignment = await prisma.topicAssignment.create({
     *   data: {
     *     // ... data to create a TopicAssignment
     *   }
     * })
     * 
     */
    create<T extends TopicAssignmentCreateArgs>(args: SelectSubset<T, TopicAssignmentCreateArgs<ExtArgs>>): Prisma__TopicAssignmentClient<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many TopicAssignments.
     * @param {TopicAssignmentCreateManyArgs} args - Arguments to create many TopicAssignments.
     * @example
     * // Create many TopicAssignments
     * const topicAssignment = await prisma.topicAssignment.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends TopicAssignmentCreateManyArgs>(args?: SelectSubset<T, TopicAssignmentCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many TopicAssignments and returns the data saved in the database.
     * @param {TopicAssignmentCreateManyAndReturnArgs} args - Arguments to create many TopicAssignments.
     * @example
     * // Create many TopicAssignments
     * const topicAssignment = await prisma.topicAssignment.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many TopicAssignments and only return the `id`
     * const topicAssignmentWithIdOnly = await prisma.topicAssignment.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends TopicAssignmentCreateManyAndReturnArgs>(args?: SelectSubset<T, TopicAssignmentCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a TopicAssignment.
     * @param {TopicAssignmentDeleteArgs} args - Arguments to delete one TopicAssignment.
     * @example
     * // Delete one TopicAssignment
     * const TopicAssignment = await prisma.topicAssignment.delete({
     *   where: {
     *     // ... filter to delete one TopicAssignment
     *   }
     * })
     * 
     */
    delete<T extends TopicAssignmentDeleteArgs>(args: SelectSubset<T, TopicAssignmentDeleteArgs<ExtArgs>>): Prisma__TopicAssignmentClient<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one TopicAssignment.
     * @param {TopicAssignmentUpdateArgs} args - Arguments to update one TopicAssignment.
     * @example
     * // Update one TopicAssignment
     * const topicAssignment = await prisma.topicAssignment.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends TopicAssignmentUpdateArgs>(args: SelectSubset<T, TopicAssignmentUpdateArgs<ExtArgs>>): Prisma__TopicAssignmentClient<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more TopicAssignments.
     * @param {TopicAssignmentDeleteManyArgs} args - Arguments to filter TopicAssignments to delete.
     * @example
     * // Delete a few TopicAssignments
     * const { count } = await prisma.topicAssignment.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends TopicAssignmentDeleteManyArgs>(args?: SelectSubset<T, TopicAssignmentDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more TopicAssignments.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicAssignmentUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many TopicAssignments
     * const topicAssignment = await prisma.topicAssignment.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends TopicAssignmentUpdateManyArgs>(args: SelectSubset<T, TopicAssignmentUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more TopicAssignments and returns the data updated in the database.
     * @param {TopicAssignmentUpdateManyAndReturnArgs} args - Arguments to update many TopicAssignments.
     * @example
     * // Update many TopicAssignments
     * const topicAssignment = await prisma.topicAssignment.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more TopicAssignments and only return the `id`
     * const topicAssignmentWithIdOnly = await prisma.topicAssignment.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends TopicAssignmentUpdateManyAndReturnArgs>(args: SelectSubset<T, TopicAssignmentUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one TopicAssignment.
     * @param {TopicAssignmentUpsertArgs} args - Arguments to update or create a TopicAssignment.
     * @example
     * // Update or create a TopicAssignment
     * const topicAssignment = await prisma.topicAssignment.upsert({
     *   create: {
     *     // ... data to create a TopicAssignment
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the TopicAssignment we want to update
     *   }
     * })
     */
    upsert<T extends TopicAssignmentUpsertArgs>(args: SelectSubset<T, TopicAssignmentUpsertArgs<ExtArgs>>): Prisma__TopicAssignmentClient<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of TopicAssignments.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicAssignmentCountArgs} args - Arguments to filter TopicAssignments to count.
     * @example
     * // Count the number of TopicAssignments
     * const count = await prisma.topicAssignment.count({
     *   where: {
     *     // ... the filter for the TopicAssignments we want to count
     *   }
     * })
    **/
    count<T extends TopicAssignmentCountArgs>(
      args?: Subset<T, TopicAssignmentCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], TopicAssignmentCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a TopicAssignment.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicAssignmentAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends TopicAssignmentAggregateArgs>(args: Subset<T, TopicAssignmentAggregateArgs>): Prisma.PrismaPromise<GetTopicAssignmentAggregateType<T>>

    /**
     * Group by TopicAssignment.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {TopicAssignmentGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends TopicAssignmentGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: TopicAssignmentGroupByArgs['orderBy'] }
        : { orderBy?: TopicAssignmentGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, TopicAssignmentGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetTopicAssignmentGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the TopicAssignment model
   */
  readonly fields: TopicAssignmentFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for TopicAssignment.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__TopicAssignmentClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    question<T extends QuestionDefaultArgs<ExtArgs> = {}>(args?: Subset<T, QuestionDefaultArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    topic<T extends TopicDefaultArgs<ExtArgs> = {}>(args?: Subset<T, TopicDefaultArgs<ExtArgs>>): Prisma__TopicClient<$Result.GetResult<Prisma.$TopicPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    analysisRun<T extends AnalysisRunDefaultArgs<ExtArgs> = {}>(args?: Subset<T, AnalysisRunDefaultArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the TopicAssignment model
   */
  interface TopicAssignmentFieldRefs {
    readonly id: FieldRef<"TopicAssignment", 'String'>
    readonly questionId: FieldRef<"TopicAssignment", 'String'>
    readonly topicId: FieldRef<"TopicAssignment", 'String'>
    readonly similarityScore: FieldRef<"TopicAssignment", 'Float'>
    readonly assignmentType: FieldRef<"TopicAssignment", 'AssignmentType'>
    readonly confidence: FieldRef<"TopicAssignment", 'Float'>
    readonly analysisRunId: FieldRef<"TopicAssignment", 'String'>
    readonly createdAt: FieldRef<"TopicAssignment", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * TopicAssignment findUnique
   */
  export type TopicAssignmentFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    /**
     * Filter, which TopicAssignment to fetch.
     */
    where: TopicAssignmentWhereUniqueInput
  }

  /**
   * TopicAssignment findUniqueOrThrow
   */
  export type TopicAssignmentFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    /**
     * Filter, which TopicAssignment to fetch.
     */
    where: TopicAssignmentWhereUniqueInput
  }

  /**
   * TopicAssignment findFirst
   */
  export type TopicAssignmentFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    /**
     * Filter, which TopicAssignment to fetch.
     */
    where?: TopicAssignmentWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of TopicAssignments to fetch.
     */
    orderBy?: TopicAssignmentOrderByWithRelationInput | TopicAssignmentOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for TopicAssignments.
     */
    cursor?: TopicAssignmentWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` TopicAssignments from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` TopicAssignments.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of TopicAssignments.
     */
    distinct?: TopicAssignmentScalarFieldEnum | TopicAssignmentScalarFieldEnum[]
  }

  /**
   * TopicAssignment findFirstOrThrow
   */
  export type TopicAssignmentFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    /**
     * Filter, which TopicAssignment to fetch.
     */
    where?: TopicAssignmentWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of TopicAssignments to fetch.
     */
    orderBy?: TopicAssignmentOrderByWithRelationInput | TopicAssignmentOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for TopicAssignments.
     */
    cursor?: TopicAssignmentWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` TopicAssignments from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` TopicAssignments.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of TopicAssignments.
     */
    distinct?: TopicAssignmentScalarFieldEnum | TopicAssignmentScalarFieldEnum[]
  }

  /**
   * TopicAssignment findMany
   */
  export type TopicAssignmentFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    /**
     * Filter, which TopicAssignments to fetch.
     */
    where?: TopicAssignmentWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of TopicAssignments to fetch.
     */
    orderBy?: TopicAssignmentOrderByWithRelationInput | TopicAssignmentOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing TopicAssignments.
     */
    cursor?: TopicAssignmentWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` TopicAssignments from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` TopicAssignments.
     */
    skip?: number
    distinct?: TopicAssignmentScalarFieldEnum | TopicAssignmentScalarFieldEnum[]
  }

  /**
   * TopicAssignment create
   */
  export type TopicAssignmentCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    /**
     * The data needed to create a TopicAssignment.
     */
    data: XOR<TopicAssignmentCreateInput, TopicAssignmentUncheckedCreateInput>
  }

  /**
   * TopicAssignment createMany
   */
  export type TopicAssignmentCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many TopicAssignments.
     */
    data: TopicAssignmentCreateManyInput | TopicAssignmentCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * TopicAssignment createManyAndReturn
   */
  export type TopicAssignmentCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * The data used to create many TopicAssignments.
     */
    data: TopicAssignmentCreateManyInput | TopicAssignmentCreateManyInput[]
    skipDuplicates?: boolean
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentIncludeCreateManyAndReturn<ExtArgs> | null
  }

  /**
   * TopicAssignment update
   */
  export type TopicAssignmentUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    /**
     * The data needed to update a TopicAssignment.
     */
    data: XOR<TopicAssignmentUpdateInput, TopicAssignmentUncheckedUpdateInput>
    /**
     * Choose, which TopicAssignment to update.
     */
    where: TopicAssignmentWhereUniqueInput
  }

  /**
   * TopicAssignment updateMany
   */
  export type TopicAssignmentUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update TopicAssignments.
     */
    data: XOR<TopicAssignmentUpdateManyMutationInput, TopicAssignmentUncheckedUpdateManyInput>
    /**
     * Filter which TopicAssignments to update
     */
    where?: TopicAssignmentWhereInput
    /**
     * Limit how many TopicAssignments to update.
     */
    limit?: number
  }

  /**
   * TopicAssignment updateManyAndReturn
   */
  export type TopicAssignmentUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * The data used to update TopicAssignments.
     */
    data: XOR<TopicAssignmentUpdateManyMutationInput, TopicAssignmentUncheckedUpdateManyInput>
    /**
     * Filter which TopicAssignments to update
     */
    where?: TopicAssignmentWhereInput
    /**
     * Limit how many TopicAssignments to update.
     */
    limit?: number
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentIncludeUpdateManyAndReturn<ExtArgs> | null
  }

  /**
   * TopicAssignment upsert
   */
  export type TopicAssignmentUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    /**
     * The filter to search for the TopicAssignment to update in case it exists.
     */
    where: TopicAssignmentWhereUniqueInput
    /**
     * In case the TopicAssignment found by the `where` argument doesn't exist, create a new TopicAssignment with this data.
     */
    create: XOR<TopicAssignmentCreateInput, TopicAssignmentUncheckedCreateInput>
    /**
     * In case the TopicAssignment was found with the provided `where` argument, update it with this data.
     */
    update: XOR<TopicAssignmentUpdateInput, TopicAssignmentUncheckedUpdateInput>
  }

  /**
   * TopicAssignment delete
   */
  export type TopicAssignmentDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    /**
     * Filter which TopicAssignment to delete.
     */
    where: TopicAssignmentWhereUniqueInput
  }

  /**
   * TopicAssignment deleteMany
   */
  export type TopicAssignmentDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which TopicAssignments to delete
     */
    where?: TopicAssignmentWhereInput
    /**
     * Limit how many TopicAssignments to delete.
     */
    limit?: number
  }

  /**
   * TopicAssignment without action
   */
  export type TopicAssignmentDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
  }


  /**
   * Model ClusterResult
   */

  export type AggregateClusterResult = {
    _count: ClusterResultCountAggregateOutputType | null
    _avg: ClusterResultAvgAggregateOutputType | null
    _sum: ClusterResultSumAggregateOutputType | null
    _min: ClusterResultMinAggregateOutputType | null
    _max: ClusterResultMaxAggregateOutputType | null
  }

  export type ClusterResultAvgAggregateOutputType = {
    clusterId: number | null
  }

  export type ClusterResultSumAggregateOutputType = {
    clusterId: number | null
  }

  export type ClusterResultMinAggregateOutputType = {
    id: string | null
    questionId: string | null
    clusterId: number | null
    clusterName: string | null
    analysisRunId: string | null
    createdAt: Date | null
  }

  export type ClusterResultMaxAggregateOutputType = {
    id: string | null
    questionId: string | null
    clusterId: number | null
    clusterName: string | null
    analysisRunId: string | null
    createdAt: Date | null
  }

  export type ClusterResultCountAggregateOutputType = {
    id: number
    questionId: number
    clusterId: number
    clusterName: number
    analysisRunId: number
    createdAt: number
    _all: number
  }


  export type ClusterResultAvgAggregateInputType = {
    clusterId?: true
  }

  export type ClusterResultSumAggregateInputType = {
    clusterId?: true
  }

  export type ClusterResultMinAggregateInputType = {
    id?: true
    questionId?: true
    clusterId?: true
    clusterName?: true
    analysisRunId?: true
    createdAt?: true
  }

  export type ClusterResultMaxAggregateInputType = {
    id?: true
    questionId?: true
    clusterId?: true
    clusterName?: true
    analysisRunId?: true
    createdAt?: true
  }

  export type ClusterResultCountAggregateInputType = {
    id?: true
    questionId?: true
    clusterId?: true
    clusterName?: true
    analysisRunId?: true
    createdAt?: true
    _all?: true
  }

  export type ClusterResultAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which ClusterResult to aggregate.
     */
    where?: ClusterResultWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ClusterResults to fetch.
     */
    orderBy?: ClusterResultOrderByWithRelationInput | ClusterResultOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: ClusterResultWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ClusterResults from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ClusterResults.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned ClusterResults
    **/
    _count?: true | ClusterResultCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: ClusterResultAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: ClusterResultSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: ClusterResultMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: ClusterResultMaxAggregateInputType
  }

  export type GetClusterResultAggregateType<T extends ClusterResultAggregateArgs> = {
        [P in keyof T & keyof AggregateClusterResult]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateClusterResult[P]>
      : GetScalarType<T[P], AggregateClusterResult[P]>
  }




  export type ClusterResultGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: ClusterResultWhereInput
    orderBy?: ClusterResultOrderByWithAggregationInput | ClusterResultOrderByWithAggregationInput[]
    by: ClusterResultScalarFieldEnum[] | ClusterResultScalarFieldEnum
    having?: ClusterResultScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: ClusterResultCountAggregateInputType | true
    _avg?: ClusterResultAvgAggregateInputType
    _sum?: ClusterResultSumAggregateInputType
    _min?: ClusterResultMinAggregateInputType
    _max?: ClusterResultMaxAggregateInputType
  }

  export type ClusterResultGroupByOutputType = {
    id: string
    questionId: string
    clusterId: number
    clusterName: string | null
    analysisRunId: string
    createdAt: Date
    _count: ClusterResultCountAggregateOutputType | null
    _avg: ClusterResultAvgAggregateOutputType | null
    _sum: ClusterResultSumAggregateOutputType | null
    _min: ClusterResultMinAggregateOutputType | null
    _max: ClusterResultMaxAggregateOutputType | null
  }

  type GetClusterResultGroupByPayload<T extends ClusterResultGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<ClusterResultGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof ClusterResultGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], ClusterResultGroupByOutputType[P]>
            : GetScalarType<T[P], ClusterResultGroupByOutputType[P]>
        }
      >
    >


  export type ClusterResultSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    questionId?: boolean
    clusterId?: boolean
    clusterName?: boolean
    analysisRunId?: boolean
    createdAt?: boolean
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["clusterResult"]>

  export type ClusterResultSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    questionId?: boolean
    clusterId?: boolean
    clusterName?: boolean
    analysisRunId?: boolean
    createdAt?: boolean
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["clusterResult"]>

  export type ClusterResultSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    questionId?: boolean
    clusterId?: boolean
    clusterName?: boolean
    analysisRunId?: boolean
    createdAt?: boolean
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["clusterResult"]>

  export type ClusterResultSelectScalar = {
    id?: boolean
    questionId?: boolean
    clusterId?: boolean
    clusterName?: boolean
    analysisRunId?: boolean
    createdAt?: boolean
  }

  export type ClusterResultOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "questionId" | "clusterId" | "clusterName" | "analysisRunId" | "createdAt", ExtArgs["result"]["clusterResult"]>
  export type ClusterResultInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }
  export type ClusterResultIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }
  export type ClusterResultIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    question?: boolean | QuestionDefaultArgs<ExtArgs>
    analysisRun?: boolean | AnalysisRunDefaultArgs<ExtArgs>
  }

  export type $ClusterResultPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "ClusterResult"
    objects: {
      question: Prisma.$QuestionPayload<ExtArgs>
      analysisRun: Prisma.$AnalysisRunPayload<ExtArgs>
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      questionId: string
      clusterId: number
      clusterName: string | null
      analysisRunId: string
      createdAt: Date
    }, ExtArgs["result"]["clusterResult"]>
    composites: {}
  }

  type ClusterResultGetPayload<S extends boolean | null | undefined | ClusterResultDefaultArgs> = $Result.GetResult<Prisma.$ClusterResultPayload, S>

  type ClusterResultCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<ClusterResultFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: ClusterResultCountAggregateInputType | true
    }

  export interface ClusterResultDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['ClusterResult'], meta: { name: 'ClusterResult' } }
    /**
     * Find zero or one ClusterResult that matches the filter.
     * @param {ClusterResultFindUniqueArgs} args - Arguments to find a ClusterResult
     * @example
     * // Get one ClusterResult
     * const clusterResult = await prisma.clusterResult.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends ClusterResultFindUniqueArgs>(args: SelectSubset<T, ClusterResultFindUniqueArgs<ExtArgs>>): Prisma__ClusterResultClient<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one ClusterResult that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {ClusterResultFindUniqueOrThrowArgs} args - Arguments to find a ClusterResult
     * @example
     * // Get one ClusterResult
     * const clusterResult = await prisma.clusterResult.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends ClusterResultFindUniqueOrThrowArgs>(args: SelectSubset<T, ClusterResultFindUniqueOrThrowArgs<ExtArgs>>): Prisma__ClusterResultClient<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first ClusterResult that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ClusterResultFindFirstArgs} args - Arguments to find a ClusterResult
     * @example
     * // Get one ClusterResult
     * const clusterResult = await prisma.clusterResult.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends ClusterResultFindFirstArgs>(args?: SelectSubset<T, ClusterResultFindFirstArgs<ExtArgs>>): Prisma__ClusterResultClient<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first ClusterResult that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ClusterResultFindFirstOrThrowArgs} args - Arguments to find a ClusterResult
     * @example
     * // Get one ClusterResult
     * const clusterResult = await prisma.clusterResult.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends ClusterResultFindFirstOrThrowArgs>(args?: SelectSubset<T, ClusterResultFindFirstOrThrowArgs<ExtArgs>>): Prisma__ClusterResultClient<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more ClusterResults that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ClusterResultFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all ClusterResults
     * const clusterResults = await prisma.clusterResult.findMany()
     * 
     * // Get first 10 ClusterResults
     * const clusterResults = await prisma.clusterResult.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const clusterResultWithIdOnly = await prisma.clusterResult.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends ClusterResultFindManyArgs>(args?: SelectSubset<T, ClusterResultFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a ClusterResult.
     * @param {ClusterResultCreateArgs} args - Arguments to create a ClusterResult.
     * @example
     * // Create one ClusterResult
     * const ClusterResult = await prisma.clusterResult.create({
     *   data: {
     *     // ... data to create a ClusterResult
     *   }
     * })
     * 
     */
    create<T extends ClusterResultCreateArgs>(args: SelectSubset<T, ClusterResultCreateArgs<ExtArgs>>): Prisma__ClusterResultClient<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many ClusterResults.
     * @param {ClusterResultCreateManyArgs} args - Arguments to create many ClusterResults.
     * @example
     * // Create many ClusterResults
     * const clusterResult = await prisma.clusterResult.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends ClusterResultCreateManyArgs>(args?: SelectSubset<T, ClusterResultCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many ClusterResults and returns the data saved in the database.
     * @param {ClusterResultCreateManyAndReturnArgs} args - Arguments to create many ClusterResults.
     * @example
     * // Create many ClusterResults
     * const clusterResult = await prisma.clusterResult.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many ClusterResults and only return the `id`
     * const clusterResultWithIdOnly = await prisma.clusterResult.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends ClusterResultCreateManyAndReturnArgs>(args?: SelectSubset<T, ClusterResultCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a ClusterResult.
     * @param {ClusterResultDeleteArgs} args - Arguments to delete one ClusterResult.
     * @example
     * // Delete one ClusterResult
     * const ClusterResult = await prisma.clusterResult.delete({
     *   where: {
     *     // ... filter to delete one ClusterResult
     *   }
     * })
     * 
     */
    delete<T extends ClusterResultDeleteArgs>(args: SelectSubset<T, ClusterResultDeleteArgs<ExtArgs>>): Prisma__ClusterResultClient<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one ClusterResult.
     * @param {ClusterResultUpdateArgs} args - Arguments to update one ClusterResult.
     * @example
     * // Update one ClusterResult
     * const clusterResult = await prisma.clusterResult.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends ClusterResultUpdateArgs>(args: SelectSubset<T, ClusterResultUpdateArgs<ExtArgs>>): Prisma__ClusterResultClient<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more ClusterResults.
     * @param {ClusterResultDeleteManyArgs} args - Arguments to filter ClusterResults to delete.
     * @example
     * // Delete a few ClusterResults
     * const { count } = await prisma.clusterResult.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends ClusterResultDeleteManyArgs>(args?: SelectSubset<T, ClusterResultDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more ClusterResults.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ClusterResultUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many ClusterResults
     * const clusterResult = await prisma.clusterResult.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends ClusterResultUpdateManyArgs>(args: SelectSubset<T, ClusterResultUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more ClusterResults and returns the data updated in the database.
     * @param {ClusterResultUpdateManyAndReturnArgs} args - Arguments to update many ClusterResults.
     * @example
     * // Update many ClusterResults
     * const clusterResult = await prisma.clusterResult.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more ClusterResults and only return the `id`
     * const clusterResultWithIdOnly = await prisma.clusterResult.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends ClusterResultUpdateManyAndReturnArgs>(args: SelectSubset<T, ClusterResultUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one ClusterResult.
     * @param {ClusterResultUpsertArgs} args - Arguments to update or create a ClusterResult.
     * @example
     * // Update or create a ClusterResult
     * const clusterResult = await prisma.clusterResult.upsert({
     *   create: {
     *     // ... data to create a ClusterResult
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the ClusterResult we want to update
     *   }
     * })
     */
    upsert<T extends ClusterResultUpsertArgs>(args: SelectSubset<T, ClusterResultUpsertArgs<ExtArgs>>): Prisma__ClusterResultClient<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of ClusterResults.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ClusterResultCountArgs} args - Arguments to filter ClusterResults to count.
     * @example
     * // Count the number of ClusterResults
     * const count = await prisma.clusterResult.count({
     *   where: {
     *     // ... the filter for the ClusterResults we want to count
     *   }
     * })
    **/
    count<T extends ClusterResultCountArgs>(
      args?: Subset<T, ClusterResultCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], ClusterResultCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a ClusterResult.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ClusterResultAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends ClusterResultAggregateArgs>(args: Subset<T, ClusterResultAggregateArgs>): Prisma.PrismaPromise<GetClusterResultAggregateType<T>>

    /**
     * Group by ClusterResult.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ClusterResultGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends ClusterResultGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: ClusterResultGroupByArgs['orderBy'] }
        : { orderBy?: ClusterResultGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, ClusterResultGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetClusterResultGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the ClusterResult model
   */
  readonly fields: ClusterResultFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for ClusterResult.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__ClusterResultClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    question<T extends QuestionDefaultArgs<ExtArgs> = {}>(args?: Subset<T, QuestionDefaultArgs<ExtArgs>>): Prisma__QuestionClient<$Result.GetResult<Prisma.$QuestionPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    analysisRun<T extends AnalysisRunDefaultArgs<ExtArgs> = {}>(args?: Subset<T, AnalysisRunDefaultArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the ClusterResult model
   */
  interface ClusterResultFieldRefs {
    readonly id: FieldRef<"ClusterResult", 'String'>
    readonly questionId: FieldRef<"ClusterResult", 'String'>
    readonly clusterId: FieldRef<"ClusterResult", 'Int'>
    readonly clusterName: FieldRef<"ClusterResult", 'String'>
    readonly analysisRunId: FieldRef<"ClusterResult", 'String'>
    readonly createdAt: FieldRef<"ClusterResult", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * ClusterResult findUnique
   */
  export type ClusterResultFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    /**
     * Filter, which ClusterResult to fetch.
     */
    where: ClusterResultWhereUniqueInput
  }

  /**
   * ClusterResult findUniqueOrThrow
   */
  export type ClusterResultFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    /**
     * Filter, which ClusterResult to fetch.
     */
    where: ClusterResultWhereUniqueInput
  }

  /**
   * ClusterResult findFirst
   */
  export type ClusterResultFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    /**
     * Filter, which ClusterResult to fetch.
     */
    where?: ClusterResultWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ClusterResults to fetch.
     */
    orderBy?: ClusterResultOrderByWithRelationInput | ClusterResultOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for ClusterResults.
     */
    cursor?: ClusterResultWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ClusterResults from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ClusterResults.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of ClusterResults.
     */
    distinct?: ClusterResultScalarFieldEnum | ClusterResultScalarFieldEnum[]
  }

  /**
   * ClusterResult findFirstOrThrow
   */
  export type ClusterResultFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    /**
     * Filter, which ClusterResult to fetch.
     */
    where?: ClusterResultWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ClusterResults to fetch.
     */
    orderBy?: ClusterResultOrderByWithRelationInput | ClusterResultOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for ClusterResults.
     */
    cursor?: ClusterResultWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ClusterResults from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ClusterResults.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of ClusterResults.
     */
    distinct?: ClusterResultScalarFieldEnum | ClusterResultScalarFieldEnum[]
  }

  /**
   * ClusterResult findMany
   */
  export type ClusterResultFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    /**
     * Filter, which ClusterResults to fetch.
     */
    where?: ClusterResultWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ClusterResults to fetch.
     */
    orderBy?: ClusterResultOrderByWithRelationInput | ClusterResultOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing ClusterResults.
     */
    cursor?: ClusterResultWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ClusterResults from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ClusterResults.
     */
    skip?: number
    distinct?: ClusterResultScalarFieldEnum | ClusterResultScalarFieldEnum[]
  }

  /**
   * ClusterResult create
   */
  export type ClusterResultCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    /**
     * The data needed to create a ClusterResult.
     */
    data: XOR<ClusterResultCreateInput, ClusterResultUncheckedCreateInput>
  }

  /**
   * ClusterResult createMany
   */
  export type ClusterResultCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many ClusterResults.
     */
    data: ClusterResultCreateManyInput | ClusterResultCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * ClusterResult createManyAndReturn
   */
  export type ClusterResultCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * The data used to create many ClusterResults.
     */
    data: ClusterResultCreateManyInput | ClusterResultCreateManyInput[]
    skipDuplicates?: boolean
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultIncludeCreateManyAndReturn<ExtArgs> | null
  }

  /**
   * ClusterResult update
   */
  export type ClusterResultUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    /**
     * The data needed to update a ClusterResult.
     */
    data: XOR<ClusterResultUpdateInput, ClusterResultUncheckedUpdateInput>
    /**
     * Choose, which ClusterResult to update.
     */
    where: ClusterResultWhereUniqueInput
  }

  /**
   * ClusterResult updateMany
   */
  export type ClusterResultUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update ClusterResults.
     */
    data: XOR<ClusterResultUpdateManyMutationInput, ClusterResultUncheckedUpdateManyInput>
    /**
     * Filter which ClusterResults to update
     */
    where?: ClusterResultWhereInput
    /**
     * Limit how many ClusterResults to update.
     */
    limit?: number
  }

  /**
   * ClusterResult updateManyAndReturn
   */
  export type ClusterResultUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * The data used to update ClusterResults.
     */
    data: XOR<ClusterResultUpdateManyMutationInput, ClusterResultUncheckedUpdateManyInput>
    /**
     * Filter which ClusterResults to update
     */
    where?: ClusterResultWhereInput
    /**
     * Limit how many ClusterResults to update.
     */
    limit?: number
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultIncludeUpdateManyAndReturn<ExtArgs> | null
  }

  /**
   * ClusterResult upsert
   */
  export type ClusterResultUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    /**
     * The filter to search for the ClusterResult to update in case it exists.
     */
    where: ClusterResultWhereUniqueInput
    /**
     * In case the ClusterResult found by the `where` argument doesn't exist, create a new ClusterResult with this data.
     */
    create: XOR<ClusterResultCreateInput, ClusterResultUncheckedCreateInput>
    /**
     * In case the ClusterResult was found with the provided `where` argument, update it with this data.
     */
    update: XOR<ClusterResultUpdateInput, ClusterResultUncheckedUpdateInput>
  }

  /**
   * ClusterResult delete
   */
  export type ClusterResultDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    /**
     * Filter which ClusterResult to delete.
     */
    where: ClusterResultWhereUniqueInput
  }

  /**
   * ClusterResult deleteMany
   */
  export type ClusterResultDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which ClusterResults to delete
     */
    where?: ClusterResultWhereInput
    /**
     * Limit how many ClusterResults to delete.
     */
    limit?: number
  }

  /**
   * ClusterResult without action
   */
  export type ClusterResultDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
  }


  /**
   * Model AnalysisRun
   */

  export type AggregateAnalysisRun = {
    _count: AnalysisRunCountAggregateOutputType | null
    _avg: AnalysisRunAvgAggregateOutputType | null
    _sum: AnalysisRunSumAggregateOutputType | null
    _min: AnalysisRunMinAggregateOutputType | null
    _max: AnalysisRunMaxAggregateOutputType | null
  }

  export type AnalysisRunAvgAggregateOutputType = {
    sampleSize: number | null
    similarityThreshold: number | null
    totalQuestions: number | null
    matchedQuestions: number | null
    newTopicsFound: number | null
    processingTimeMs: number | null
  }

  export type AnalysisRunSumAggregateOutputType = {
    sampleSize: number | null
    similarityThreshold: number | null
    totalQuestions: number | null
    matchedQuestions: number | null
    newTopicsFound: number | null
    processingTimeMs: number | null
  }

  export type AnalysisRunMinAggregateOutputType = {
    id: string | null
    status: $Enums.AnalysisStatus | null
    mode: string | null
    sampleSize: number | null
    similarityThreshold: number | null
    embeddingModel: string | null
    gptModel: string | null
    totalQuestions: number | null
    matchedQuestions: number | null
    newTopicsFound: number | null
    processingTimeMs: number | null
    errorMessage: string | null
    startedAt: Date | null
    completedAt: Date | null
    createdBy: string | null
  }

  export type AnalysisRunMaxAggregateOutputType = {
    id: string | null
    status: $Enums.AnalysisStatus | null
    mode: string | null
    sampleSize: number | null
    similarityThreshold: number | null
    embeddingModel: string | null
    gptModel: string | null
    totalQuestions: number | null
    matchedQuestions: number | null
    newTopicsFound: number | null
    processingTimeMs: number | null
    errorMessage: string | null
    startedAt: Date | null
    completedAt: Date | null
    createdBy: string | null
  }

  export type AnalysisRunCountAggregateOutputType = {
    id: number
    status: number
    mode: number
    sampleSize: number
    similarityThreshold: number
    embeddingModel: number
    gptModel: number
    totalQuestions: number
    matchedQuestions: number
    newTopicsFound: number
    processingTimeMs: number
    errorMessage: number
    startedAt: number
    completedAt: number
    createdBy: number
    config: number
    _all: number
  }


  export type AnalysisRunAvgAggregateInputType = {
    sampleSize?: true
    similarityThreshold?: true
    totalQuestions?: true
    matchedQuestions?: true
    newTopicsFound?: true
    processingTimeMs?: true
  }

  export type AnalysisRunSumAggregateInputType = {
    sampleSize?: true
    similarityThreshold?: true
    totalQuestions?: true
    matchedQuestions?: true
    newTopicsFound?: true
    processingTimeMs?: true
  }

  export type AnalysisRunMinAggregateInputType = {
    id?: true
    status?: true
    mode?: true
    sampleSize?: true
    similarityThreshold?: true
    embeddingModel?: true
    gptModel?: true
    totalQuestions?: true
    matchedQuestions?: true
    newTopicsFound?: true
    processingTimeMs?: true
    errorMessage?: true
    startedAt?: true
    completedAt?: true
    createdBy?: true
  }

  export type AnalysisRunMaxAggregateInputType = {
    id?: true
    status?: true
    mode?: true
    sampleSize?: true
    similarityThreshold?: true
    embeddingModel?: true
    gptModel?: true
    totalQuestions?: true
    matchedQuestions?: true
    newTopicsFound?: true
    processingTimeMs?: true
    errorMessage?: true
    startedAt?: true
    completedAt?: true
    createdBy?: true
  }

  export type AnalysisRunCountAggregateInputType = {
    id?: true
    status?: true
    mode?: true
    sampleSize?: true
    similarityThreshold?: true
    embeddingModel?: true
    gptModel?: true
    totalQuestions?: true
    matchedQuestions?: true
    newTopicsFound?: true
    processingTimeMs?: true
    errorMessage?: true
    startedAt?: true
    completedAt?: true
    createdBy?: true
    config?: true
    _all?: true
  }

  export type AnalysisRunAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which AnalysisRun to aggregate.
     */
    where?: AnalysisRunWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of AnalysisRuns to fetch.
     */
    orderBy?: AnalysisRunOrderByWithRelationInput | AnalysisRunOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: AnalysisRunWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` AnalysisRuns from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` AnalysisRuns.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned AnalysisRuns
    **/
    _count?: true | AnalysisRunCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: AnalysisRunAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: AnalysisRunSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: AnalysisRunMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: AnalysisRunMaxAggregateInputType
  }

  export type GetAnalysisRunAggregateType<T extends AnalysisRunAggregateArgs> = {
        [P in keyof T & keyof AggregateAnalysisRun]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateAnalysisRun[P]>
      : GetScalarType<T[P], AggregateAnalysisRun[P]>
  }




  export type AnalysisRunGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: AnalysisRunWhereInput
    orderBy?: AnalysisRunOrderByWithAggregationInput | AnalysisRunOrderByWithAggregationInput[]
    by: AnalysisRunScalarFieldEnum[] | AnalysisRunScalarFieldEnum
    having?: AnalysisRunScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: AnalysisRunCountAggregateInputType | true
    _avg?: AnalysisRunAvgAggregateInputType
    _sum?: AnalysisRunSumAggregateInputType
    _min?: AnalysisRunMinAggregateInputType
    _max?: AnalysisRunMaxAggregateInputType
  }

  export type AnalysisRunGroupByOutputType = {
    id: string
    status: $Enums.AnalysisStatus
    mode: string
    sampleSize: number | null
    similarityThreshold: number
    embeddingModel: string
    gptModel: string
    totalQuestions: number | null
    matchedQuestions: number | null
    newTopicsFound: number | null
    processingTimeMs: number | null
    errorMessage: string | null
    startedAt: Date
    completedAt: Date | null
    createdBy: string | null
    config: JsonValue | null
    _count: AnalysisRunCountAggregateOutputType | null
    _avg: AnalysisRunAvgAggregateOutputType | null
    _sum: AnalysisRunSumAggregateOutputType | null
    _min: AnalysisRunMinAggregateOutputType | null
    _max: AnalysisRunMaxAggregateOutputType | null
  }

  type GetAnalysisRunGroupByPayload<T extends AnalysisRunGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<AnalysisRunGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof AnalysisRunGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], AnalysisRunGroupByOutputType[P]>
            : GetScalarType<T[P], AnalysisRunGroupByOutputType[P]>
        }
      >
    >


  export type AnalysisRunSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    status?: boolean
    mode?: boolean
    sampleSize?: boolean
    similarityThreshold?: boolean
    embeddingModel?: boolean
    gptModel?: boolean
    totalQuestions?: boolean
    matchedQuestions?: boolean
    newTopicsFound?: boolean
    processingTimeMs?: boolean
    errorMessage?: boolean
    startedAt?: boolean
    completedAt?: boolean
    createdBy?: boolean
    config?: boolean
    topicAssignments?: boolean | AnalysisRun$topicAssignmentsArgs<ExtArgs>
    clusterResults?: boolean | AnalysisRun$clusterResultsArgs<ExtArgs>
    _count?: boolean | AnalysisRunCountOutputTypeDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["analysisRun"]>

  export type AnalysisRunSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    status?: boolean
    mode?: boolean
    sampleSize?: boolean
    similarityThreshold?: boolean
    embeddingModel?: boolean
    gptModel?: boolean
    totalQuestions?: boolean
    matchedQuestions?: boolean
    newTopicsFound?: boolean
    processingTimeMs?: boolean
    errorMessage?: boolean
    startedAt?: boolean
    completedAt?: boolean
    createdBy?: boolean
    config?: boolean
  }, ExtArgs["result"]["analysisRun"]>

  export type AnalysisRunSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    status?: boolean
    mode?: boolean
    sampleSize?: boolean
    similarityThreshold?: boolean
    embeddingModel?: boolean
    gptModel?: boolean
    totalQuestions?: boolean
    matchedQuestions?: boolean
    newTopicsFound?: boolean
    processingTimeMs?: boolean
    errorMessage?: boolean
    startedAt?: boolean
    completedAt?: boolean
    createdBy?: boolean
    config?: boolean
  }, ExtArgs["result"]["analysisRun"]>

  export type AnalysisRunSelectScalar = {
    id?: boolean
    status?: boolean
    mode?: boolean
    sampleSize?: boolean
    similarityThreshold?: boolean
    embeddingModel?: boolean
    gptModel?: boolean
    totalQuestions?: boolean
    matchedQuestions?: boolean
    newTopicsFound?: boolean
    processingTimeMs?: boolean
    errorMessage?: boolean
    startedAt?: boolean
    completedAt?: boolean
    createdBy?: boolean
    config?: boolean
  }

  export type AnalysisRunOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "status" | "mode" | "sampleSize" | "similarityThreshold" | "embeddingModel" | "gptModel" | "totalQuestions" | "matchedQuestions" | "newTopicsFound" | "processingTimeMs" | "errorMessage" | "startedAt" | "completedAt" | "createdBy" | "config", ExtArgs["result"]["analysisRun"]>
  export type AnalysisRunInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    topicAssignments?: boolean | AnalysisRun$topicAssignmentsArgs<ExtArgs>
    clusterResults?: boolean | AnalysisRun$clusterResultsArgs<ExtArgs>
    _count?: boolean | AnalysisRunCountOutputTypeDefaultArgs<ExtArgs>
  }
  export type AnalysisRunIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}
  export type AnalysisRunIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}

  export type $AnalysisRunPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "AnalysisRun"
    objects: {
      topicAssignments: Prisma.$TopicAssignmentPayload<ExtArgs>[]
      clusterResults: Prisma.$ClusterResultPayload<ExtArgs>[]
    }
    scalars: $Extensions.GetPayloadResult<{
      id: string
      status: $Enums.AnalysisStatus
      mode: string
      sampleSize: number | null
      similarityThreshold: number
      embeddingModel: string
      gptModel: string
      totalQuestions: number | null
      matchedQuestions: number | null
      newTopicsFound: number | null
      processingTimeMs: number | null
      errorMessage: string | null
      startedAt: Date
      completedAt: Date | null
      createdBy: string | null
      config: Prisma.JsonValue | null
    }, ExtArgs["result"]["analysisRun"]>
    composites: {}
  }

  type AnalysisRunGetPayload<S extends boolean | null | undefined | AnalysisRunDefaultArgs> = $Result.GetResult<Prisma.$AnalysisRunPayload, S>

  type AnalysisRunCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<AnalysisRunFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: AnalysisRunCountAggregateInputType | true
    }

  export interface AnalysisRunDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['AnalysisRun'], meta: { name: 'AnalysisRun' } }
    /**
     * Find zero or one AnalysisRun that matches the filter.
     * @param {AnalysisRunFindUniqueArgs} args - Arguments to find a AnalysisRun
     * @example
     * // Get one AnalysisRun
     * const analysisRun = await prisma.analysisRun.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends AnalysisRunFindUniqueArgs>(args: SelectSubset<T, AnalysisRunFindUniqueArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one AnalysisRun that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {AnalysisRunFindUniqueOrThrowArgs} args - Arguments to find a AnalysisRun
     * @example
     * // Get one AnalysisRun
     * const analysisRun = await prisma.analysisRun.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends AnalysisRunFindUniqueOrThrowArgs>(args: SelectSubset<T, AnalysisRunFindUniqueOrThrowArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first AnalysisRun that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {AnalysisRunFindFirstArgs} args - Arguments to find a AnalysisRun
     * @example
     * // Get one AnalysisRun
     * const analysisRun = await prisma.analysisRun.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends AnalysisRunFindFirstArgs>(args?: SelectSubset<T, AnalysisRunFindFirstArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first AnalysisRun that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {AnalysisRunFindFirstOrThrowArgs} args - Arguments to find a AnalysisRun
     * @example
     * // Get one AnalysisRun
     * const analysisRun = await prisma.analysisRun.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends AnalysisRunFindFirstOrThrowArgs>(args?: SelectSubset<T, AnalysisRunFindFirstOrThrowArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more AnalysisRuns that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {AnalysisRunFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all AnalysisRuns
     * const analysisRuns = await prisma.analysisRun.findMany()
     * 
     * // Get first 10 AnalysisRuns
     * const analysisRuns = await prisma.analysisRun.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const analysisRunWithIdOnly = await prisma.analysisRun.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends AnalysisRunFindManyArgs>(args?: SelectSubset<T, AnalysisRunFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a AnalysisRun.
     * @param {AnalysisRunCreateArgs} args - Arguments to create a AnalysisRun.
     * @example
     * // Create one AnalysisRun
     * const AnalysisRun = await prisma.analysisRun.create({
     *   data: {
     *     // ... data to create a AnalysisRun
     *   }
     * })
     * 
     */
    create<T extends AnalysisRunCreateArgs>(args: SelectSubset<T, AnalysisRunCreateArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many AnalysisRuns.
     * @param {AnalysisRunCreateManyArgs} args - Arguments to create many AnalysisRuns.
     * @example
     * // Create many AnalysisRuns
     * const analysisRun = await prisma.analysisRun.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends AnalysisRunCreateManyArgs>(args?: SelectSubset<T, AnalysisRunCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many AnalysisRuns and returns the data saved in the database.
     * @param {AnalysisRunCreateManyAndReturnArgs} args - Arguments to create many AnalysisRuns.
     * @example
     * // Create many AnalysisRuns
     * const analysisRun = await prisma.analysisRun.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many AnalysisRuns and only return the `id`
     * const analysisRunWithIdOnly = await prisma.analysisRun.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends AnalysisRunCreateManyAndReturnArgs>(args?: SelectSubset<T, AnalysisRunCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a AnalysisRun.
     * @param {AnalysisRunDeleteArgs} args - Arguments to delete one AnalysisRun.
     * @example
     * // Delete one AnalysisRun
     * const AnalysisRun = await prisma.analysisRun.delete({
     *   where: {
     *     // ... filter to delete one AnalysisRun
     *   }
     * })
     * 
     */
    delete<T extends AnalysisRunDeleteArgs>(args: SelectSubset<T, AnalysisRunDeleteArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one AnalysisRun.
     * @param {AnalysisRunUpdateArgs} args - Arguments to update one AnalysisRun.
     * @example
     * // Update one AnalysisRun
     * const analysisRun = await prisma.analysisRun.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends AnalysisRunUpdateArgs>(args: SelectSubset<T, AnalysisRunUpdateArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more AnalysisRuns.
     * @param {AnalysisRunDeleteManyArgs} args - Arguments to filter AnalysisRuns to delete.
     * @example
     * // Delete a few AnalysisRuns
     * const { count } = await prisma.analysisRun.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends AnalysisRunDeleteManyArgs>(args?: SelectSubset<T, AnalysisRunDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more AnalysisRuns.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {AnalysisRunUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many AnalysisRuns
     * const analysisRun = await prisma.analysisRun.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends AnalysisRunUpdateManyArgs>(args: SelectSubset<T, AnalysisRunUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more AnalysisRuns and returns the data updated in the database.
     * @param {AnalysisRunUpdateManyAndReturnArgs} args - Arguments to update many AnalysisRuns.
     * @example
     * // Update many AnalysisRuns
     * const analysisRun = await prisma.analysisRun.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more AnalysisRuns and only return the `id`
     * const analysisRunWithIdOnly = await prisma.analysisRun.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends AnalysisRunUpdateManyAndReturnArgs>(args: SelectSubset<T, AnalysisRunUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one AnalysisRun.
     * @param {AnalysisRunUpsertArgs} args - Arguments to update or create a AnalysisRun.
     * @example
     * // Update or create a AnalysisRun
     * const analysisRun = await prisma.analysisRun.upsert({
     *   create: {
     *     // ... data to create a AnalysisRun
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the AnalysisRun we want to update
     *   }
     * })
     */
    upsert<T extends AnalysisRunUpsertArgs>(args: SelectSubset<T, AnalysisRunUpsertArgs<ExtArgs>>): Prisma__AnalysisRunClient<$Result.GetResult<Prisma.$AnalysisRunPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of AnalysisRuns.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {AnalysisRunCountArgs} args - Arguments to filter AnalysisRuns to count.
     * @example
     * // Count the number of AnalysisRuns
     * const count = await prisma.analysisRun.count({
     *   where: {
     *     // ... the filter for the AnalysisRuns we want to count
     *   }
     * })
    **/
    count<T extends AnalysisRunCountArgs>(
      args?: Subset<T, AnalysisRunCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], AnalysisRunCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a AnalysisRun.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {AnalysisRunAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends AnalysisRunAggregateArgs>(args: Subset<T, AnalysisRunAggregateArgs>): Prisma.PrismaPromise<GetAnalysisRunAggregateType<T>>

    /**
     * Group by AnalysisRun.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {AnalysisRunGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends AnalysisRunGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: AnalysisRunGroupByArgs['orderBy'] }
        : { orderBy?: AnalysisRunGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, AnalysisRunGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetAnalysisRunGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the AnalysisRun model
   */
  readonly fields: AnalysisRunFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for AnalysisRun.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__AnalysisRunClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    topicAssignments<T extends AnalysisRun$topicAssignmentsArgs<ExtArgs> = {}>(args?: Subset<T, AnalysisRun$topicAssignmentsArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$TopicAssignmentPayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    clusterResults<T extends AnalysisRun$clusterResultsArgs<ExtArgs> = {}>(args?: Subset<T, AnalysisRun$clusterResultsArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ClusterResultPayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the AnalysisRun model
   */
  interface AnalysisRunFieldRefs {
    readonly id: FieldRef<"AnalysisRun", 'String'>
    readonly status: FieldRef<"AnalysisRun", 'AnalysisStatus'>
    readonly mode: FieldRef<"AnalysisRun", 'String'>
    readonly sampleSize: FieldRef<"AnalysisRun", 'Int'>
    readonly similarityThreshold: FieldRef<"AnalysisRun", 'Float'>
    readonly embeddingModel: FieldRef<"AnalysisRun", 'String'>
    readonly gptModel: FieldRef<"AnalysisRun", 'String'>
    readonly totalQuestions: FieldRef<"AnalysisRun", 'Int'>
    readonly matchedQuestions: FieldRef<"AnalysisRun", 'Int'>
    readonly newTopicsFound: FieldRef<"AnalysisRun", 'Int'>
    readonly processingTimeMs: FieldRef<"AnalysisRun", 'Int'>
    readonly errorMessage: FieldRef<"AnalysisRun", 'String'>
    readonly startedAt: FieldRef<"AnalysisRun", 'DateTime'>
    readonly completedAt: FieldRef<"AnalysisRun", 'DateTime'>
    readonly createdBy: FieldRef<"AnalysisRun", 'String'>
    readonly config: FieldRef<"AnalysisRun", 'Json'>
  }
    

  // Custom InputTypes
  /**
   * AnalysisRun findUnique
   */
  export type AnalysisRunFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
    /**
     * Filter, which AnalysisRun to fetch.
     */
    where: AnalysisRunWhereUniqueInput
  }

  /**
   * AnalysisRun findUniqueOrThrow
   */
  export type AnalysisRunFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
    /**
     * Filter, which AnalysisRun to fetch.
     */
    where: AnalysisRunWhereUniqueInput
  }

  /**
   * AnalysisRun findFirst
   */
  export type AnalysisRunFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
    /**
     * Filter, which AnalysisRun to fetch.
     */
    where?: AnalysisRunWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of AnalysisRuns to fetch.
     */
    orderBy?: AnalysisRunOrderByWithRelationInput | AnalysisRunOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for AnalysisRuns.
     */
    cursor?: AnalysisRunWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` AnalysisRuns from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` AnalysisRuns.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of AnalysisRuns.
     */
    distinct?: AnalysisRunScalarFieldEnum | AnalysisRunScalarFieldEnum[]
  }

  /**
   * AnalysisRun findFirstOrThrow
   */
  export type AnalysisRunFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
    /**
     * Filter, which AnalysisRun to fetch.
     */
    where?: AnalysisRunWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of AnalysisRuns to fetch.
     */
    orderBy?: AnalysisRunOrderByWithRelationInput | AnalysisRunOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for AnalysisRuns.
     */
    cursor?: AnalysisRunWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` AnalysisRuns from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` AnalysisRuns.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of AnalysisRuns.
     */
    distinct?: AnalysisRunScalarFieldEnum | AnalysisRunScalarFieldEnum[]
  }

  /**
   * AnalysisRun findMany
   */
  export type AnalysisRunFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
    /**
     * Filter, which AnalysisRuns to fetch.
     */
    where?: AnalysisRunWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of AnalysisRuns to fetch.
     */
    orderBy?: AnalysisRunOrderByWithRelationInput | AnalysisRunOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing AnalysisRuns.
     */
    cursor?: AnalysisRunWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` AnalysisRuns from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` AnalysisRuns.
     */
    skip?: number
    distinct?: AnalysisRunScalarFieldEnum | AnalysisRunScalarFieldEnum[]
  }

  /**
   * AnalysisRun create
   */
  export type AnalysisRunCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
    /**
     * The data needed to create a AnalysisRun.
     */
    data: XOR<AnalysisRunCreateInput, AnalysisRunUncheckedCreateInput>
  }

  /**
   * AnalysisRun createMany
   */
  export type AnalysisRunCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many AnalysisRuns.
     */
    data: AnalysisRunCreateManyInput | AnalysisRunCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * AnalysisRun createManyAndReturn
   */
  export type AnalysisRunCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * The data used to create many AnalysisRuns.
     */
    data: AnalysisRunCreateManyInput | AnalysisRunCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * AnalysisRun update
   */
  export type AnalysisRunUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
    /**
     * The data needed to update a AnalysisRun.
     */
    data: XOR<AnalysisRunUpdateInput, AnalysisRunUncheckedUpdateInput>
    /**
     * Choose, which AnalysisRun to update.
     */
    where: AnalysisRunWhereUniqueInput
  }

  /**
   * AnalysisRun updateMany
   */
  export type AnalysisRunUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update AnalysisRuns.
     */
    data: XOR<AnalysisRunUpdateManyMutationInput, AnalysisRunUncheckedUpdateManyInput>
    /**
     * Filter which AnalysisRuns to update
     */
    where?: AnalysisRunWhereInput
    /**
     * Limit how many AnalysisRuns to update.
     */
    limit?: number
  }

  /**
   * AnalysisRun updateManyAndReturn
   */
  export type AnalysisRunUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * The data used to update AnalysisRuns.
     */
    data: XOR<AnalysisRunUpdateManyMutationInput, AnalysisRunUncheckedUpdateManyInput>
    /**
     * Filter which AnalysisRuns to update
     */
    where?: AnalysisRunWhereInput
    /**
     * Limit how many AnalysisRuns to update.
     */
    limit?: number
  }

  /**
   * AnalysisRun upsert
   */
  export type AnalysisRunUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
    /**
     * The filter to search for the AnalysisRun to update in case it exists.
     */
    where: AnalysisRunWhereUniqueInput
    /**
     * In case the AnalysisRun found by the `where` argument doesn't exist, create a new AnalysisRun with this data.
     */
    create: XOR<AnalysisRunCreateInput, AnalysisRunUncheckedCreateInput>
    /**
     * In case the AnalysisRun was found with the provided `where` argument, update it with this data.
     */
    update: XOR<AnalysisRunUpdateInput, AnalysisRunUncheckedUpdateInput>
  }

  /**
   * AnalysisRun delete
   */
  export type AnalysisRunDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
    /**
     * Filter which AnalysisRun to delete.
     */
    where: AnalysisRunWhereUniqueInput
  }

  /**
   * AnalysisRun deleteMany
   */
  export type AnalysisRunDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which AnalysisRuns to delete
     */
    where?: AnalysisRunWhereInput
    /**
     * Limit how many AnalysisRuns to delete.
     */
    limit?: number
  }

  /**
   * AnalysisRun.topicAssignments
   */
  export type AnalysisRun$topicAssignmentsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the TopicAssignment
     */
    select?: TopicAssignmentSelect<ExtArgs> | null
    /**
     * Omit specific fields from the TopicAssignment
     */
    omit?: TopicAssignmentOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: TopicAssignmentInclude<ExtArgs> | null
    where?: TopicAssignmentWhereInput
    orderBy?: TopicAssignmentOrderByWithRelationInput | TopicAssignmentOrderByWithRelationInput[]
    cursor?: TopicAssignmentWhereUniqueInput
    take?: number
    skip?: number
    distinct?: TopicAssignmentScalarFieldEnum | TopicAssignmentScalarFieldEnum[]
  }

  /**
   * AnalysisRun.clusterResults
   */
  export type AnalysisRun$clusterResultsArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ClusterResult
     */
    select?: ClusterResultSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ClusterResult
     */
    omit?: ClusterResultOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ClusterResultInclude<ExtArgs> | null
    where?: ClusterResultWhereInput
    orderBy?: ClusterResultOrderByWithRelationInput | ClusterResultOrderByWithRelationInput[]
    cursor?: ClusterResultWhereUniqueInput
    take?: number
    skip?: number
    distinct?: ClusterResultScalarFieldEnum | ClusterResultScalarFieldEnum[]
  }

  /**
   * AnalysisRun without action
   */
  export type AnalysisRunDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the AnalysisRun
     */
    select?: AnalysisRunSelect<ExtArgs> | null
    /**
     * Omit specific fields from the AnalysisRun
     */
    omit?: AnalysisRunOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: AnalysisRunInclude<ExtArgs> | null
  }


  /**
   * Model SystemConfig
   */

  export type AggregateSystemConfig = {
    _count: SystemConfigCountAggregateOutputType | null
    _min: SystemConfigMinAggregateOutputType | null
    _max: SystemConfigMaxAggregateOutputType | null
  }

  export type SystemConfigMinAggregateOutputType = {
    id: string | null
    key: string | null
    value: string | null
    type: $Enums.ConfigType | null
    updatedAt: Date | null
    updatedBy: string | null
  }

  export type SystemConfigMaxAggregateOutputType = {
    id: string | null
    key: string | null
    value: string | null
    type: $Enums.ConfigType | null
    updatedAt: Date | null
    updatedBy: string | null
  }

  export type SystemConfigCountAggregateOutputType = {
    id: number
    key: number
    value: number
    type: number
    updatedAt: number
    updatedBy: number
    _all: number
  }


  export type SystemConfigMinAggregateInputType = {
    id?: true
    key?: true
    value?: true
    type?: true
    updatedAt?: true
    updatedBy?: true
  }

  export type SystemConfigMaxAggregateInputType = {
    id?: true
    key?: true
    value?: true
    type?: true
    updatedAt?: true
    updatedBy?: true
  }

  export type SystemConfigCountAggregateInputType = {
    id?: true
    key?: true
    value?: true
    type?: true
    updatedAt?: true
    updatedBy?: true
    _all?: true
  }

  export type SystemConfigAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which SystemConfig to aggregate.
     */
    where?: SystemConfigWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of SystemConfigs to fetch.
     */
    orderBy?: SystemConfigOrderByWithRelationInput | SystemConfigOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: SystemConfigWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` SystemConfigs from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` SystemConfigs.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned SystemConfigs
    **/
    _count?: true | SystemConfigCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: SystemConfigMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: SystemConfigMaxAggregateInputType
  }

  export type GetSystemConfigAggregateType<T extends SystemConfigAggregateArgs> = {
        [P in keyof T & keyof AggregateSystemConfig]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateSystemConfig[P]>
      : GetScalarType<T[P], AggregateSystemConfig[P]>
  }




  export type SystemConfigGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: SystemConfigWhereInput
    orderBy?: SystemConfigOrderByWithAggregationInput | SystemConfigOrderByWithAggregationInput[]
    by: SystemConfigScalarFieldEnum[] | SystemConfigScalarFieldEnum
    having?: SystemConfigScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: SystemConfigCountAggregateInputType | true
    _min?: SystemConfigMinAggregateInputType
    _max?: SystemConfigMaxAggregateInputType
  }

  export type SystemConfigGroupByOutputType = {
    id: string
    key: string
    value: string
    type: $Enums.ConfigType
    updatedAt: Date
    updatedBy: string | null
    _count: SystemConfigCountAggregateOutputType | null
    _min: SystemConfigMinAggregateOutputType | null
    _max: SystemConfigMaxAggregateOutputType | null
  }

  type GetSystemConfigGroupByPayload<T extends SystemConfigGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<SystemConfigGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof SystemConfigGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], SystemConfigGroupByOutputType[P]>
            : GetScalarType<T[P], SystemConfigGroupByOutputType[P]>
        }
      >
    >


  export type SystemConfigSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    key?: boolean
    value?: boolean
    type?: boolean
    updatedAt?: boolean
    updatedBy?: boolean
  }, ExtArgs["result"]["systemConfig"]>

  export type SystemConfigSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    key?: boolean
    value?: boolean
    type?: boolean
    updatedAt?: boolean
    updatedBy?: boolean
  }, ExtArgs["result"]["systemConfig"]>

  export type SystemConfigSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    key?: boolean
    value?: boolean
    type?: boolean
    updatedAt?: boolean
    updatedBy?: boolean
  }, ExtArgs["result"]["systemConfig"]>

  export type SystemConfigSelectScalar = {
    id?: boolean
    key?: boolean
    value?: boolean
    type?: boolean
    updatedAt?: boolean
    updatedBy?: boolean
  }

  export type SystemConfigOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "key" | "value" | "type" | "updatedAt" | "updatedBy", ExtArgs["result"]["systemConfig"]>

  export type $SystemConfigPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "SystemConfig"
    objects: {}
    scalars: $Extensions.GetPayloadResult<{
      id: string
      key: string
      value: string
      type: $Enums.ConfigType
      updatedAt: Date
      updatedBy: string | null
    }, ExtArgs["result"]["systemConfig"]>
    composites: {}
  }

  type SystemConfigGetPayload<S extends boolean | null | undefined | SystemConfigDefaultArgs> = $Result.GetResult<Prisma.$SystemConfigPayload, S>

  type SystemConfigCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<SystemConfigFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: SystemConfigCountAggregateInputType | true
    }

  export interface SystemConfigDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['SystemConfig'], meta: { name: 'SystemConfig' } }
    /**
     * Find zero or one SystemConfig that matches the filter.
     * @param {SystemConfigFindUniqueArgs} args - Arguments to find a SystemConfig
     * @example
     * // Get one SystemConfig
     * const systemConfig = await prisma.systemConfig.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends SystemConfigFindUniqueArgs>(args: SelectSubset<T, SystemConfigFindUniqueArgs<ExtArgs>>): Prisma__SystemConfigClient<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one SystemConfig that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {SystemConfigFindUniqueOrThrowArgs} args - Arguments to find a SystemConfig
     * @example
     * // Get one SystemConfig
     * const systemConfig = await prisma.systemConfig.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends SystemConfigFindUniqueOrThrowArgs>(args: SelectSubset<T, SystemConfigFindUniqueOrThrowArgs<ExtArgs>>): Prisma__SystemConfigClient<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first SystemConfig that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {SystemConfigFindFirstArgs} args - Arguments to find a SystemConfig
     * @example
     * // Get one SystemConfig
     * const systemConfig = await prisma.systemConfig.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends SystemConfigFindFirstArgs>(args?: SelectSubset<T, SystemConfigFindFirstArgs<ExtArgs>>): Prisma__SystemConfigClient<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first SystemConfig that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {SystemConfigFindFirstOrThrowArgs} args - Arguments to find a SystemConfig
     * @example
     * // Get one SystemConfig
     * const systemConfig = await prisma.systemConfig.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends SystemConfigFindFirstOrThrowArgs>(args?: SelectSubset<T, SystemConfigFindFirstOrThrowArgs<ExtArgs>>): Prisma__SystemConfigClient<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more SystemConfigs that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {SystemConfigFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all SystemConfigs
     * const systemConfigs = await prisma.systemConfig.findMany()
     * 
     * // Get first 10 SystemConfigs
     * const systemConfigs = await prisma.systemConfig.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const systemConfigWithIdOnly = await prisma.systemConfig.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends SystemConfigFindManyArgs>(args?: SelectSubset<T, SystemConfigFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a SystemConfig.
     * @param {SystemConfigCreateArgs} args - Arguments to create a SystemConfig.
     * @example
     * // Create one SystemConfig
     * const SystemConfig = await prisma.systemConfig.create({
     *   data: {
     *     // ... data to create a SystemConfig
     *   }
     * })
     * 
     */
    create<T extends SystemConfigCreateArgs>(args: SelectSubset<T, SystemConfigCreateArgs<ExtArgs>>): Prisma__SystemConfigClient<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many SystemConfigs.
     * @param {SystemConfigCreateManyArgs} args - Arguments to create many SystemConfigs.
     * @example
     * // Create many SystemConfigs
     * const systemConfig = await prisma.systemConfig.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends SystemConfigCreateManyArgs>(args?: SelectSubset<T, SystemConfigCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many SystemConfigs and returns the data saved in the database.
     * @param {SystemConfigCreateManyAndReturnArgs} args - Arguments to create many SystemConfigs.
     * @example
     * // Create many SystemConfigs
     * const systemConfig = await prisma.systemConfig.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many SystemConfigs and only return the `id`
     * const systemConfigWithIdOnly = await prisma.systemConfig.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends SystemConfigCreateManyAndReturnArgs>(args?: SelectSubset<T, SystemConfigCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a SystemConfig.
     * @param {SystemConfigDeleteArgs} args - Arguments to delete one SystemConfig.
     * @example
     * // Delete one SystemConfig
     * const SystemConfig = await prisma.systemConfig.delete({
     *   where: {
     *     // ... filter to delete one SystemConfig
     *   }
     * })
     * 
     */
    delete<T extends SystemConfigDeleteArgs>(args: SelectSubset<T, SystemConfigDeleteArgs<ExtArgs>>): Prisma__SystemConfigClient<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one SystemConfig.
     * @param {SystemConfigUpdateArgs} args - Arguments to update one SystemConfig.
     * @example
     * // Update one SystemConfig
     * const systemConfig = await prisma.systemConfig.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends SystemConfigUpdateArgs>(args: SelectSubset<T, SystemConfigUpdateArgs<ExtArgs>>): Prisma__SystemConfigClient<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more SystemConfigs.
     * @param {SystemConfigDeleteManyArgs} args - Arguments to filter SystemConfigs to delete.
     * @example
     * // Delete a few SystemConfigs
     * const { count } = await prisma.systemConfig.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends SystemConfigDeleteManyArgs>(args?: SelectSubset<T, SystemConfigDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more SystemConfigs.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {SystemConfigUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many SystemConfigs
     * const systemConfig = await prisma.systemConfig.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends SystemConfigUpdateManyArgs>(args: SelectSubset<T, SystemConfigUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more SystemConfigs and returns the data updated in the database.
     * @param {SystemConfigUpdateManyAndReturnArgs} args - Arguments to update many SystemConfigs.
     * @example
     * // Update many SystemConfigs
     * const systemConfig = await prisma.systemConfig.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more SystemConfigs and only return the `id`
     * const systemConfigWithIdOnly = await prisma.systemConfig.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends SystemConfigUpdateManyAndReturnArgs>(args: SelectSubset<T, SystemConfigUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one SystemConfig.
     * @param {SystemConfigUpsertArgs} args - Arguments to update or create a SystemConfig.
     * @example
     * // Update or create a SystemConfig
     * const systemConfig = await prisma.systemConfig.upsert({
     *   create: {
     *     // ... data to create a SystemConfig
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the SystemConfig we want to update
     *   }
     * })
     */
    upsert<T extends SystemConfigUpsertArgs>(args: SelectSubset<T, SystemConfigUpsertArgs<ExtArgs>>): Prisma__SystemConfigClient<$Result.GetResult<Prisma.$SystemConfigPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of SystemConfigs.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {SystemConfigCountArgs} args - Arguments to filter SystemConfigs to count.
     * @example
     * // Count the number of SystemConfigs
     * const count = await prisma.systemConfig.count({
     *   where: {
     *     // ... the filter for the SystemConfigs we want to count
     *   }
     * })
    **/
    count<T extends SystemConfigCountArgs>(
      args?: Subset<T, SystemConfigCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], SystemConfigCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a SystemConfig.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {SystemConfigAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends SystemConfigAggregateArgs>(args: Subset<T, SystemConfigAggregateArgs>): Prisma.PrismaPromise<GetSystemConfigAggregateType<T>>

    /**
     * Group by SystemConfig.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {SystemConfigGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends SystemConfigGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: SystemConfigGroupByArgs['orderBy'] }
        : { orderBy?: SystemConfigGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, SystemConfigGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetSystemConfigGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the SystemConfig model
   */
  readonly fields: SystemConfigFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for SystemConfig.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__SystemConfigClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the SystemConfig model
   */
  interface SystemConfigFieldRefs {
    readonly id: FieldRef<"SystemConfig", 'String'>
    readonly key: FieldRef<"SystemConfig", 'String'>
    readonly value: FieldRef<"SystemConfig", 'String'>
    readonly type: FieldRef<"SystemConfig", 'ConfigType'>
    readonly updatedAt: FieldRef<"SystemConfig", 'DateTime'>
    readonly updatedBy: FieldRef<"SystemConfig", 'String'>
  }
    

  // Custom InputTypes
  /**
   * SystemConfig findUnique
   */
  export type SystemConfigFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * Filter, which SystemConfig to fetch.
     */
    where: SystemConfigWhereUniqueInput
  }

  /**
   * SystemConfig findUniqueOrThrow
   */
  export type SystemConfigFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * Filter, which SystemConfig to fetch.
     */
    where: SystemConfigWhereUniqueInput
  }

  /**
   * SystemConfig findFirst
   */
  export type SystemConfigFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * Filter, which SystemConfig to fetch.
     */
    where?: SystemConfigWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of SystemConfigs to fetch.
     */
    orderBy?: SystemConfigOrderByWithRelationInput | SystemConfigOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for SystemConfigs.
     */
    cursor?: SystemConfigWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` SystemConfigs from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` SystemConfigs.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of SystemConfigs.
     */
    distinct?: SystemConfigScalarFieldEnum | SystemConfigScalarFieldEnum[]
  }

  /**
   * SystemConfig findFirstOrThrow
   */
  export type SystemConfigFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * Filter, which SystemConfig to fetch.
     */
    where?: SystemConfigWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of SystemConfigs to fetch.
     */
    orderBy?: SystemConfigOrderByWithRelationInput | SystemConfigOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for SystemConfigs.
     */
    cursor?: SystemConfigWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` SystemConfigs from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` SystemConfigs.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of SystemConfigs.
     */
    distinct?: SystemConfigScalarFieldEnum | SystemConfigScalarFieldEnum[]
  }

  /**
   * SystemConfig findMany
   */
  export type SystemConfigFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * Filter, which SystemConfigs to fetch.
     */
    where?: SystemConfigWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of SystemConfigs to fetch.
     */
    orderBy?: SystemConfigOrderByWithRelationInput | SystemConfigOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing SystemConfigs.
     */
    cursor?: SystemConfigWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` SystemConfigs from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` SystemConfigs.
     */
    skip?: number
    distinct?: SystemConfigScalarFieldEnum | SystemConfigScalarFieldEnum[]
  }

  /**
   * SystemConfig create
   */
  export type SystemConfigCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * The data needed to create a SystemConfig.
     */
    data: XOR<SystemConfigCreateInput, SystemConfigUncheckedCreateInput>
  }

  /**
   * SystemConfig createMany
   */
  export type SystemConfigCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many SystemConfigs.
     */
    data: SystemConfigCreateManyInput | SystemConfigCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * SystemConfig createManyAndReturn
   */
  export type SystemConfigCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * The data used to create many SystemConfigs.
     */
    data: SystemConfigCreateManyInput | SystemConfigCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * SystemConfig update
   */
  export type SystemConfigUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * The data needed to update a SystemConfig.
     */
    data: XOR<SystemConfigUpdateInput, SystemConfigUncheckedUpdateInput>
    /**
     * Choose, which SystemConfig to update.
     */
    where: SystemConfigWhereUniqueInput
  }

  /**
   * SystemConfig updateMany
   */
  export type SystemConfigUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update SystemConfigs.
     */
    data: XOR<SystemConfigUpdateManyMutationInput, SystemConfigUncheckedUpdateManyInput>
    /**
     * Filter which SystemConfigs to update
     */
    where?: SystemConfigWhereInput
    /**
     * Limit how many SystemConfigs to update.
     */
    limit?: number
  }

  /**
   * SystemConfig updateManyAndReturn
   */
  export type SystemConfigUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * The data used to update SystemConfigs.
     */
    data: XOR<SystemConfigUpdateManyMutationInput, SystemConfigUncheckedUpdateManyInput>
    /**
     * Filter which SystemConfigs to update
     */
    where?: SystemConfigWhereInput
    /**
     * Limit how many SystemConfigs to update.
     */
    limit?: number
  }

  /**
   * SystemConfig upsert
   */
  export type SystemConfigUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * The filter to search for the SystemConfig to update in case it exists.
     */
    where: SystemConfigWhereUniqueInput
    /**
     * In case the SystemConfig found by the `where` argument doesn't exist, create a new SystemConfig with this data.
     */
    create: XOR<SystemConfigCreateInput, SystemConfigUncheckedCreateInput>
    /**
     * In case the SystemConfig was found with the provided `where` argument, update it with this data.
     */
    update: XOR<SystemConfigUpdateInput, SystemConfigUncheckedUpdateInput>
  }

  /**
   * SystemConfig delete
   */
  export type SystemConfigDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
    /**
     * Filter which SystemConfig to delete.
     */
    where: SystemConfigWhereUniqueInput
  }

  /**
   * SystemConfig deleteMany
   */
  export type SystemConfigDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which SystemConfigs to delete
     */
    where?: SystemConfigWhereInput
    /**
     * Limit how many SystemConfigs to delete.
     */
    limit?: number
  }

  /**
   * SystemConfig without action
   */
  export type SystemConfigDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the SystemConfig
     */
    select?: SystemConfigSelect<ExtArgs> | null
    /**
     * Omit specific fields from the SystemConfig
     */
    omit?: SystemConfigOmit<ExtArgs> | null
  }


  /**
   * Model UserSession
   */

  export type AggregateUserSession = {
    _count: UserSessionCountAggregateOutputType | null
    _min: UserSessionMinAggregateOutputType | null
    _max: UserSessionMaxAggregateOutputType | null
  }

  export type UserSessionMinAggregateOutputType = {
    id: string | null
    sessionId: string | null
    userId: string | null
    isDevSession: boolean | null
    ipAddress: string | null
    userAgent: string | null
    country: string | null
    region: string | null
    createdAt: Date | null
    lastActivity: Date | null
  }

  export type UserSessionMaxAggregateOutputType = {
    id: string | null
    sessionId: string | null
    userId: string | null
    isDevSession: boolean | null
    ipAddress: string | null
    userAgent: string | null
    country: string | null
    region: string | null
    createdAt: Date | null
    lastActivity: Date | null
  }

  export type UserSessionCountAggregateOutputType = {
    id: number
    sessionId: number
    userId: number
    isDevSession: number
    ipAddress: number
    userAgent: number
    country: number
    region: number
    createdAt: number
    lastActivity: number
    _all: number
  }


  export type UserSessionMinAggregateInputType = {
    id?: true
    sessionId?: true
    userId?: true
    isDevSession?: true
    ipAddress?: true
    userAgent?: true
    country?: true
    region?: true
    createdAt?: true
    lastActivity?: true
  }

  export type UserSessionMaxAggregateInputType = {
    id?: true
    sessionId?: true
    userId?: true
    isDevSession?: true
    ipAddress?: true
    userAgent?: true
    country?: true
    region?: true
    createdAt?: true
    lastActivity?: true
  }

  export type UserSessionCountAggregateInputType = {
    id?: true
    sessionId?: true
    userId?: true
    isDevSession?: true
    ipAddress?: true
    userAgent?: true
    country?: true
    region?: true
    createdAt?: true
    lastActivity?: true
    _all?: true
  }

  export type UserSessionAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which UserSession to aggregate.
     */
    where?: UserSessionWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of UserSessions to fetch.
     */
    orderBy?: UserSessionOrderByWithRelationInput | UserSessionOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: UserSessionWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` UserSessions from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` UserSessions.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned UserSessions
    **/
    _count?: true | UserSessionCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: UserSessionMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: UserSessionMaxAggregateInputType
  }

  export type GetUserSessionAggregateType<T extends UserSessionAggregateArgs> = {
        [P in keyof T & keyof AggregateUserSession]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateUserSession[P]>
      : GetScalarType<T[P], AggregateUserSession[P]>
  }




  export type UserSessionGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: UserSessionWhereInput
    orderBy?: UserSessionOrderByWithAggregationInput | UserSessionOrderByWithAggregationInput[]
    by: UserSessionScalarFieldEnum[] | UserSessionScalarFieldEnum
    having?: UserSessionScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: UserSessionCountAggregateInputType | true
    _min?: UserSessionMinAggregateInputType
    _max?: UserSessionMaxAggregateInputType
  }

  export type UserSessionGroupByOutputType = {
    id: string
    sessionId: string
    userId: string | null
    isDevSession: boolean
    ipAddress: string | null
    userAgent: string | null
    country: string | null
    region: string | null
    createdAt: Date
    lastActivity: Date
    _count: UserSessionCountAggregateOutputType | null
    _min: UserSessionMinAggregateOutputType | null
    _max: UserSessionMaxAggregateOutputType | null
  }

  type GetUserSessionGroupByPayload<T extends UserSessionGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<UserSessionGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof UserSessionGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], UserSessionGroupByOutputType[P]>
            : GetScalarType<T[P], UserSessionGroupByOutputType[P]>
        }
      >
    >


  export type UserSessionSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    sessionId?: boolean
    userId?: boolean
    isDevSession?: boolean
    ipAddress?: boolean
    userAgent?: boolean
    country?: boolean
    region?: boolean
    createdAt?: boolean
    lastActivity?: boolean
  }, ExtArgs["result"]["userSession"]>

  export type UserSessionSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    sessionId?: boolean
    userId?: boolean
    isDevSession?: boolean
    ipAddress?: boolean
    userAgent?: boolean
    country?: boolean
    region?: boolean
    createdAt?: boolean
    lastActivity?: boolean
  }, ExtArgs["result"]["userSession"]>

  export type UserSessionSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    sessionId?: boolean
    userId?: boolean
    isDevSession?: boolean
    ipAddress?: boolean
    userAgent?: boolean
    country?: boolean
    region?: boolean
    createdAt?: boolean
    lastActivity?: boolean
  }, ExtArgs["result"]["userSession"]>

  export type UserSessionSelectScalar = {
    id?: boolean
    sessionId?: boolean
    userId?: boolean
    isDevSession?: boolean
    ipAddress?: boolean
    userAgent?: boolean
    country?: boolean
    region?: boolean
    createdAt?: boolean
    lastActivity?: boolean
  }

  export type UserSessionOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "sessionId" | "userId" | "isDevSession" | "ipAddress" | "userAgent" | "country" | "region" | "createdAt" | "lastActivity", ExtArgs["result"]["userSession"]>

  export type $UserSessionPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "UserSession"
    objects: {}
    scalars: $Extensions.GetPayloadResult<{
      id: string
      sessionId: string
      userId: string | null
      isDevSession: boolean
      ipAddress: string | null
      userAgent: string | null
      country: string | null
      region: string | null
      createdAt: Date
      lastActivity: Date
    }, ExtArgs["result"]["userSession"]>
    composites: {}
  }

  type UserSessionGetPayload<S extends boolean | null | undefined | UserSessionDefaultArgs> = $Result.GetResult<Prisma.$UserSessionPayload, S>

  type UserSessionCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<UserSessionFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: UserSessionCountAggregateInputType | true
    }

  export interface UserSessionDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['UserSession'], meta: { name: 'UserSession' } }
    /**
     * Find zero or one UserSession that matches the filter.
     * @param {UserSessionFindUniqueArgs} args - Arguments to find a UserSession
     * @example
     * // Get one UserSession
     * const userSession = await prisma.userSession.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends UserSessionFindUniqueArgs>(args: SelectSubset<T, UserSessionFindUniqueArgs<ExtArgs>>): Prisma__UserSessionClient<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one UserSession that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {UserSessionFindUniqueOrThrowArgs} args - Arguments to find a UserSession
     * @example
     * // Get one UserSession
     * const userSession = await prisma.userSession.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends UserSessionFindUniqueOrThrowArgs>(args: SelectSubset<T, UserSessionFindUniqueOrThrowArgs<ExtArgs>>): Prisma__UserSessionClient<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first UserSession that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserSessionFindFirstArgs} args - Arguments to find a UserSession
     * @example
     * // Get one UserSession
     * const userSession = await prisma.userSession.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends UserSessionFindFirstArgs>(args?: SelectSubset<T, UserSessionFindFirstArgs<ExtArgs>>): Prisma__UserSessionClient<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first UserSession that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserSessionFindFirstOrThrowArgs} args - Arguments to find a UserSession
     * @example
     * // Get one UserSession
     * const userSession = await prisma.userSession.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends UserSessionFindFirstOrThrowArgs>(args?: SelectSubset<T, UserSessionFindFirstOrThrowArgs<ExtArgs>>): Prisma__UserSessionClient<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more UserSessions that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserSessionFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all UserSessions
     * const userSessions = await prisma.userSession.findMany()
     * 
     * // Get first 10 UserSessions
     * const userSessions = await prisma.userSession.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const userSessionWithIdOnly = await prisma.userSession.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends UserSessionFindManyArgs>(args?: SelectSubset<T, UserSessionFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a UserSession.
     * @param {UserSessionCreateArgs} args - Arguments to create a UserSession.
     * @example
     * // Create one UserSession
     * const UserSession = await prisma.userSession.create({
     *   data: {
     *     // ... data to create a UserSession
     *   }
     * })
     * 
     */
    create<T extends UserSessionCreateArgs>(args: SelectSubset<T, UserSessionCreateArgs<ExtArgs>>): Prisma__UserSessionClient<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many UserSessions.
     * @param {UserSessionCreateManyArgs} args - Arguments to create many UserSessions.
     * @example
     * // Create many UserSessions
     * const userSession = await prisma.userSession.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends UserSessionCreateManyArgs>(args?: SelectSubset<T, UserSessionCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many UserSessions and returns the data saved in the database.
     * @param {UserSessionCreateManyAndReturnArgs} args - Arguments to create many UserSessions.
     * @example
     * // Create many UserSessions
     * const userSession = await prisma.userSession.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many UserSessions and only return the `id`
     * const userSessionWithIdOnly = await prisma.userSession.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends UserSessionCreateManyAndReturnArgs>(args?: SelectSubset<T, UserSessionCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a UserSession.
     * @param {UserSessionDeleteArgs} args - Arguments to delete one UserSession.
     * @example
     * // Delete one UserSession
     * const UserSession = await prisma.userSession.delete({
     *   where: {
     *     // ... filter to delete one UserSession
     *   }
     * })
     * 
     */
    delete<T extends UserSessionDeleteArgs>(args: SelectSubset<T, UserSessionDeleteArgs<ExtArgs>>): Prisma__UserSessionClient<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one UserSession.
     * @param {UserSessionUpdateArgs} args - Arguments to update one UserSession.
     * @example
     * // Update one UserSession
     * const userSession = await prisma.userSession.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends UserSessionUpdateArgs>(args: SelectSubset<T, UserSessionUpdateArgs<ExtArgs>>): Prisma__UserSessionClient<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more UserSessions.
     * @param {UserSessionDeleteManyArgs} args - Arguments to filter UserSessions to delete.
     * @example
     * // Delete a few UserSessions
     * const { count } = await prisma.userSession.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends UserSessionDeleteManyArgs>(args?: SelectSubset<T, UserSessionDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more UserSessions.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserSessionUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many UserSessions
     * const userSession = await prisma.userSession.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends UserSessionUpdateManyArgs>(args: SelectSubset<T, UserSessionUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more UserSessions and returns the data updated in the database.
     * @param {UserSessionUpdateManyAndReturnArgs} args - Arguments to update many UserSessions.
     * @example
     * // Update many UserSessions
     * const userSession = await prisma.userSession.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more UserSessions and only return the `id`
     * const userSessionWithIdOnly = await prisma.userSession.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends UserSessionUpdateManyAndReturnArgs>(args: SelectSubset<T, UserSessionUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one UserSession.
     * @param {UserSessionUpsertArgs} args - Arguments to update or create a UserSession.
     * @example
     * // Update or create a UserSession
     * const userSession = await prisma.userSession.upsert({
     *   create: {
     *     // ... data to create a UserSession
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the UserSession we want to update
     *   }
     * })
     */
    upsert<T extends UserSessionUpsertArgs>(args: SelectSubset<T, UserSessionUpsertArgs<ExtArgs>>): Prisma__UserSessionClient<$Result.GetResult<Prisma.$UserSessionPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of UserSessions.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserSessionCountArgs} args - Arguments to filter UserSessions to count.
     * @example
     * // Count the number of UserSessions
     * const count = await prisma.userSession.count({
     *   where: {
     *     // ... the filter for the UserSessions we want to count
     *   }
     * })
    **/
    count<T extends UserSessionCountArgs>(
      args?: Subset<T, UserSessionCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], UserSessionCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a UserSession.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserSessionAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends UserSessionAggregateArgs>(args: Subset<T, UserSessionAggregateArgs>): Prisma.PrismaPromise<GetUserSessionAggregateType<T>>

    /**
     * Group by UserSession.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserSessionGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends UserSessionGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: UserSessionGroupByArgs['orderBy'] }
        : { orderBy?: UserSessionGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, UserSessionGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetUserSessionGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the UserSession model
   */
  readonly fields: UserSessionFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for UserSession.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__UserSessionClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the UserSession model
   */
  interface UserSessionFieldRefs {
    readonly id: FieldRef<"UserSession", 'String'>
    readonly sessionId: FieldRef<"UserSession", 'String'>
    readonly userId: FieldRef<"UserSession", 'String'>
    readonly isDevSession: FieldRef<"UserSession", 'Boolean'>
    readonly ipAddress: FieldRef<"UserSession", 'String'>
    readonly userAgent: FieldRef<"UserSession", 'String'>
    readonly country: FieldRef<"UserSession", 'String'>
    readonly region: FieldRef<"UserSession", 'String'>
    readonly createdAt: FieldRef<"UserSession", 'DateTime'>
    readonly lastActivity: FieldRef<"UserSession", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * UserSession findUnique
   */
  export type UserSessionFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * Filter, which UserSession to fetch.
     */
    where: UserSessionWhereUniqueInput
  }

  /**
   * UserSession findUniqueOrThrow
   */
  export type UserSessionFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * Filter, which UserSession to fetch.
     */
    where: UserSessionWhereUniqueInput
  }

  /**
   * UserSession findFirst
   */
  export type UserSessionFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * Filter, which UserSession to fetch.
     */
    where?: UserSessionWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of UserSessions to fetch.
     */
    orderBy?: UserSessionOrderByWithRelationInput | UserSessionOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for UserSessions.
     */
    cursor?: UserSessionWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` UserSessions from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` UserSessions.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of UserSessions.
     */
    distinct?: UserSessionScalarFieldEnum | UserSessionScalarFieldEnum[]
  }

  /**
   * UserSession findFirstOrThrow
   */
  export type UserSessionFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * Filter, which UserSession to fetch.
     */
    where?: UserSessionWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of UserSessions to fetch.
     */
    orderBy?: UserSessionOrderByWithRelationInput | UserSessionOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for UserSessions.
     */
    cursor?: UserSessionWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` UserSessions from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` UserSessions.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of UserSessions.
     */
    distinct?: UserSessionScalarFieldEnum | UserSessionScalarFieldEnum[]
  }

  /**
   * UserSession findMany
   */
  export type UserSessionFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * Filter, which UserSessions to fetch.
     */
    where?: UserSessionWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of UserSessions to fetch.
     */
    orderBy?: UserSessionOrderByWithRelationInput | UserSessionOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing UserSessions.
     */
    cursor?: UserSessionWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` UserSessions from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` UserSessions.
     */
    skip?: number
    distinct?: UserSessionScalarFieldEnum | UserSessionScalarFieldEnum[]
  }

  /**
   * UserSession create
   */
  export type UserSessionCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * The data needed to create a UserSession.
     */
    data: XOR<UserSessionCreateInput, UserSessionUncheckedCreateInput>
  }

  /**
   * UserSession createMany
   */
  export type UserSessionCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many UserSessions.
     */
    data: UserSessionCreateManyInput | UserSessionCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * UserSession createManyAndReturn
   */
  export type UserSessionCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * The data used to create many UserSessions.
     */
    data: UserSessionCreateManyInput | UserSessionCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * UserSession update
   */
  export type UserSessionUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * The data needed to update a UserSession.
     */
    data: XOR<UserSessionUpdateInput, UserSessionUncheckedUpdateInput>
    /**
     * Choose, which UserSession to update.
     */
    where: UserSessionWhereUniqueInput
  }

  /**
   * UserSession updateMany
   */
  export type UserSessionUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update UserSessions.
     */
    data: XOR<UserSessionUpdateManyMutationInput, UserSessionUncheckedUpdateManyInput>
    /**
     * Filter which UserSessions to update
     */
    where?: UserSessionWhereInput
    /**
     * Limit how many UserSessions to update.
     */
    limit?: number
  }

  /**
   * UserSession updateManyAndReturn
   */
  export type UserSessionUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * The data used to update UserSessions.
     */
    data: XOR<UserSessionUpdateManyMutationInput, UserSessionUncheckedUpdateManyInput>
    /**
     * Filter which UserSessions to update
     */
    where?: UserSessionWhereInput
    /**
     * Limit how many UserSessions to update.
     */
    limit?: number
  }

  /**
   * UserSession upsert
   */
  export type UserSessionUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * The filter to search for the UserSession to update in case it exists.
     */
    where: UserSessionWhereUniqueInput
    /**
     * In case the UserSession found by the `where` argument doesn't exist, create a new UserSession with this data.
     */
    create: XOR<UserSessionCreateInput, UserSessionUncheckedCreateInput>
    /**
     * In case the UserSession was found with the provided `where` argument, update it with this data.
     */
    update: XOR<UserSessionUpdateInput, UserSessionUncheckedUpdateInput>
  }

  /**
   * UserSession delete
   */
  export type UserSessionDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
    /**
     * Filter which UserSession to delete.
     */
    where: UserSessionWhereUniqueInput
  }

  /**
   * UserSession deleteMany
   */
  export type UserSessionDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which UserSessions to delete
     */
    where?: UserSessionWhereInput
    /**
     * Limit how many UserSessions to delete.
     */
    limit?: number
  }

  /**
   * UserSession without action
   */
  export type UserSessionDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the UserSession
     */
    select?: UserSessionSelect<ExtArgs> | null
    /**
     * Omit specific fields from the UserSession
     */
    omit?: UserSessionOmit<ExtArgs> | null
  }


  /**
   * Model DataSync
   */

  export type AggregateDataSync = {
    _count: DataSyncCountAggregateOutputType | null
    _avg: DataSyncAvgAggregateOutputType | null
    _sum: DataSyncSumAggregateOutputType | null
    _min: DataSyncMinAggregateOutputType | null
    _max: DataSyncMaxAggregateOutputType | null
  }

  export type DataSyncAvgAggregateOutputType = {
    recordsProcessed: number | null
    recordsAdded: number | null
    recordsUpdated: number | null
  }

  export type DataSyncSumAggregateOutputType = {
    recordsProcessed: number | null
    recordsAdded: number | null
    recordsUpdated: number | null
  }

  export type DataSyncMinAggregateOutputType = {
    id: string | null
    source: string | null
    sourceUrl: string | null
    lastSyncAt: Date | null
    status: $Enums.SyncStatus | null
    recordsProcessed: number | null
    recordsAdded: number | null
    recordsUpdated: number | null
    errorMessage: string | null
  }

  export type DataSyncMaxAggregateOutputType = {
    id: string | null
    source: string | null
    sourceUrl: string | null
    lastSyncAt: Date | null
    status: $Enums.SyncStatus | null
    recordsProcessed: number | null
    recordsAdded: number | null
    recordsUpdated: number | null
    errorMessage: string | null
  }

  export type DataSyncCountAggregateOutputType = {
    id: number
    source: number
    sourceUrl: number
    lastSyncAt: number
    status: number
    recordsProcessed: number
    recordsAdded: number
    recordsUpdated: number
    errorMessage: number
    _all: number
  }


  export type DataSyncAvgAggregateInputType = {
    recordsProcessed?: true
    recordsAdded?: true
    recordsUpdated?: true
  }

  export type DataSyncSumAggregateInputType = {
    recordsProcessed?: true
    recordsAdded?: true
    recordsUpdated?: true
  }

  export type DataSyncMinAggregateInputType = {
    id?: true
    source?: true
    sourceUrl?: true
    lastSyncAt?: true
    status?: true
    recordsProcessed?: true
    recordsAdded?: true
    recordsUpdated?: true
    errorMessage?: true
  }

  export type DataSyncMaxAggregateInputType = {
    id?: true
    source?: true
    sourceUrl?: true
    lastSyncAt?: true
    status?: true
    recordsProcessed?: true
    recordsAdded?: true
    recordsUpdated?: true
    errorMessage?: true
  }

  export type DataSyncCountAggregateInputType = {
    id?: true
    source?: true
    sourceUrl?: true
    lastSyncAt?: true
    status?: true
    recordsProcessed?: true
    recordsAdded?: true
    recordsUpdated?: true
    errorMessage?: true
    _all?: true
  }

  export type DataSyncAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which DataSync to aggregate.
     */
    where?: DataSyncWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataSyncs to fetch.
     */
    orderBy?: DataSyncOrderByWithRelationInput | DataSyncOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: DataSyncWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataSyncs from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataSyncs.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned DataSyncs
    **/
    _count?: true | DataSyncCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: DataSyncAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: DataSyncSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: DataSyncMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: DataSyncMaxAggregateInputType
  }

  export type GetDataSyncAggregateType<T extends DataSyncAggregateArgs> = {
        [P in keyof T & keyof AggregateDataSync]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateDataSync[P]>
      : GetScalarType<T[P], AggregateDataSync[P]>
  }




  export type DataSyncGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: DataSyncWhereInput
    orderBy?: DataSyncOrderByWithAggregationInput | DataSyncOrderByWithAggregationInput[]
    by: DataSyncScalarFieldEnum[] | DataSyncScalarFieldEnum
    having?: DataSyncScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: DataSyncCountAggregateInputType | true
    _avg?: DataSyncAvgAggregateInputType
    _sum?: DataSyncSumAggregateInputType
    _min?: DataSyncMinAggregateInputType
    _max?: DataSyncMaxAggregateInputType
  }

  export type DataSyncGroupByOutputType = {
    id: string
    source: string
    sourceUrl: string | null
    lastSyncAt: Date
    status: $Enums.SyncStatus
    recordsProcessed: number | null
    recordsAdded: number | null
    recordsUpdated: number | null
    errorMessage: string | null
    _count: DataSyncCountAggregateOutputType | null
    _avg: DataSyncAvgAggregateOutputType | null
    _sum: DataSyncSumAggregateOutputType | null
    _min: DataSyncMinAggregateOutputType | null
    _max: DataSyncMaxAggregateOutputType | null
  }

  type GetDataSyncGroupByPayload<T extends DataSyncGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<DataSyncGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof DataSyncGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], DataSyncGroupByOutputType[P]>
            : GetScalarType<T[P], DataSyncGroupByOutputType[P]>
        }
      >
    >


  export type DataSyncSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    source?: boolean
    sourceUrl?: boolean
    lastSyncAt?: boolean
    status?: boolean
    recordsProcessed?: boolean
    recordsAdded?: boolean
    recordsUpdated?: boolean
    errorMessage?: boolean
  }, ExtArgs["result"]["dataSync"]>

  export type DataSyncSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    source?: boolean
    sourceUrl?: boolean
    lastSyncAt?: boolean
    status?: boolean
    recordsProcessed?: boolean
    recordsAdded?: boolean
    recordsUpdated?: boolean
    errorMessage?: boolean
  }, ExtArgs["result"]["dataSync"]>

  export type DataSyncSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    source?: boolean
    sourceUrl?: boolean
    lastSyncAt?: boolean
    status?: boolean
    recordsProcessed?: boolean
    recordsAdded?: boolean
    recordsUpdated?: boolean
    errorMessage?: boolean
  }, ExtArgs["result"]["dataSync"]>

  export type DataSyncSelectScalar = {
    id?: boolean
    source?: boolean
    sourceUrl?: boolean
    lastSyncAt?: boolean
    status?: boolean
    recordsProcessed?: boolean
    recordsAdded?: boolean
    recordsUpdated?: boolean
    errorMessage?: boolean
  }

  export type DataSyncOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "source" | "sourceUrl" | "lastSyncAt" | "status" | "recordsProcessed" | "recordsAdded" | "recordsUpdated" | "errorMessage", ExtArgs["result"]["dataSync"]>

  export type $DataSyncPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "DataSync"
    objects: {}
    scalars: $Extensions.GetPayloadResult<{
      id: string
      source: string
      sourceUrl: string | null
      lastSyncAt: Date
      status: $Enums.SyncStatus
      recordsProcessed: number | null
      recordsAdded: number | null
      recordsUpdated: number | null
      errorMessage: string | null
    }, ExtArgs["result"]["dataSync"]>
    composites: {}
  }

  type DataSyncGetPayload<S extends boolean | null | undefined | DataSyncDefaultArgs> = $Result.GetResult<Prisma.$DataSyncPayload, S>

  type DataSyncCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<DataSyncFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: DataSyncCountAggregateInputType | true
    }

  export interface DataSyncDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['DataSync'], meta: { name: 'DataSync' } }
    /**
     * Find zero or one DataSync that matches the filter.
     * @param {DataSyncFindUniqueArgs} args - Arguments to find a DataSync
     * @example
     * // Get one DataSync
     * const dataSync = await prisma.dataSync.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends DataSyncFindUniqueArgs>(args: SelectSubset<T, DataSyncFindUniqueArgs<ExtArgs>>): Prisma__DataSyncClient<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one DataSync that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {DataSyncFindUniqueOrThrowArgs} args - Arguments to find a DataSync
     * @example
     * // Get one DataSync
     * const dataSync = await prisma.dataSync.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends DataSyncFindUniqueOrThrowArgs>(args: SelectSubset<T, DataSyncFindUniqueOrThrowArgs<ExtArgs>>): Prisma__DataSyncClient<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first DataSync that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSyncFindFirstArgs} args - Arguments to find a DataSync
     * @example
     * // Get one DataSync
     * const dataSync = await prisma.dataSync.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends DataSyncFindFirstArgs>(args?: SelectSubset<T, DataSyncFindFirstArgs<ExtArgs>>): Prisma__DataSyncClient<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first DataSync that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSyncFindFirstOrThrowArgs} args - Arguments to find a DataSync
     * @example
     * // Get one DataSync
     * const dataSync = await prisma.dataSync.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends DataSyncFindFirstOrThrowArgs>(args?: SelectSubset<T, DataSyncFindFirstOrThrowArgs<ExtArgs>>): Prisma__DataSyncClient<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more DataSyncs that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSyncFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all DataSyncs
     * const dataSyncs = await prisma.dataSync.findMany()
     * 
     * // Get first 10 DataSyncs
     * const dataSyncs = await prisma.dataSync.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const dataSyncWithIdOnly = await prisma.dataSync.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends DataSyncFindManyArgs>(args?: SelectSubset<T, DataSyncFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a DataSync.
     * @param {DataSyncCreateArgs} args - Arguments to create a DataSync.
     * @example
     * // Create one DataSync
     * const DataSync = await prisma.dataSync.create({
     *   data: {
     *     // ... data to create a DataSync
     *   }
     * })
     * 
     */
    create<T extends DataSyncCreateArgs>(args: SelectSubset<T, DataSyncCreateArgs<ExtArgs>>): Prisma__DataSyncClient<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many DataSyncs.
     * @param {DataSyncCreateManyArgs} args - Arguments to create many DataSyncs.
     * @example
     * // Create many DataSyncs
     * const dataSync = await prisma.dataSync.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends DataSyncCreateManyArgs>(args?: SelectSubset<T, DataSyncCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many DataSyncs and returns the data saved in the database.
     * @param {DataSyncCreateManyAndReturnArgs} args - Arguments to create many DataSyncs.
     * @example
     * // Create many DataSyncs
     * const dataSync = await prisma.dataSync.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many DataSyncs and only return the `id`
     * const dataSyncWithIdOnly = await prisma.dataSync.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends DataSyncCreateManyAndReturnArgs>(args?: SelectSubset<T, DataSyncCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a DataSync.
     * @param {DataSyncDeleteArgs} args - Arguments to delete one DataSync.
     * @example
     * // Delete one DataSync
     * const DataSync = await prisma.dataSync.delete({
     *   where: {
     *     // ... filter to delete one DataSync
     *   }
     * })
     * 
     */
    delete<T extends DataSyncDeleteArgs>(args: SelectSubset<T, DataSyncDeleteArgs<ExtArgs>>): Prisma__DataSyncClient<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one DataSync.
     * @param {DataSyncUpdateArgs} args - Arguments to update one DataSync.
     * @example
     * // Update one DataSync
     * const dataSync = await prisma.dataSync.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends DataSyncUpdateArgs>(args: SelectSubset<T, DataSyncUpdateArgs<ExtArgs>>): Prisma__DataSyncClient<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more DataSyncs.
     * @param {DataSyncDeleteManyArgs} args - Arguments to filter DataSyncs to delete.
     * @example
     * // Delete a few DataSyncs
     * const { count } = await prisma.dataSync.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends DataSyncDeleteManyArgs>(args?: SelectSubset<T, DataSyncDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more DataSyncs.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSyncUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many DataSyncs
     * const dataSync = await prisma.dataSync.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends DataSyncUpdateManyArgs>(args: SelectSubset<T, DataSyncUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more DataSyncs and returns the data updated in the database.
     * @param {DataSyncUpdateManyAndReturnArgs} args - Arguments to update many DataSyncs.
     * @example
     * // Update many DataSyncs
     * const dataSync = await prisma.dataSync.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more DataSyncs and only return the `id`
     * const dataSyncWithIdOnly = await prisma.dataSync.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends DataSyncUpdateManyAndReturnArgs>(args: SelectSubset<T, DataSyncUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one DataSync.
     * @param {DataSyncUpsertArgs} args - Arguments to update or create a DataSync.
     * @example
     * // Update or create a DataSync
     * const dataSync = await prisma.dataSync.upsert({
     *   create: {
     *     // ... data to create a DataSync
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the DataSync we want to update
     *   }
     * })
     */
    upsert<T extends DataSyncUpsertArgs>(args: SelectSubset<T, DataSyncUpsertArgs<ExtArgs>>): Prisma__DataSyncClient<$Result.GetResult<Prisma.$DataSyncPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of DataSyncs.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSyncCountArgs} args - Arguments to filter DataSyncs to count.
     * @example
     * // Count the number of DataSyncs
     * const count = await prisma.dataSync.count({
     *   where: {
     *     // ... the filter for the DataSyncs we want to count
     *   }
     * })
    **/
    count<T extends DataSyncCountArgs>(
      args?: Subset<T, DataSyncCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], DataSyncCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a DataSync.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSyncAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends DataSyncAggregateArgs>(args: Subset<T, DataSyncAggregateArgs>): Prisma.PrismaPromise<GetDataSyncAggregateType<T>>

    /**
     * Group by DataSync.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSyncGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends DataSyncGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: DataSyncGroupByArgs['orderBy'] }
        : { orderBy?: DataSyncGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, DataSyncGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetDataSyncGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the DataSync model
   */
  readonly fields: DataSyncFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for DataSync.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__DataSyncClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the DataSync model
   */
  interface DataSyncFieldRefs {
    readonly id: FieldRef<"DataSync", 'String'>
    readonly source: FieldRef<"DataSync", 'String'>
    readonly sourceUrl: FieldRef<"DataSync", 'String'>
    readonly lastSyncAt: FieldRef<"DataSync", 'DateTime'>
    readonly status: FieldRef<"DataSync", 'SyncStatus'>
    readonly recordsProcessed: FieldRef<"DataSync", 'Int'>
    readonly recordsAdded: FieldRef<"DataSync", 'Int'>
    readonly recordsUpdated: FieldRef<"DataSync", 'Int'>
    readonly errorMessage: FieldRef<"DataSync", 'String'>
  }
    

  // Custom InputTypes
  /**
   * DataSync findUnique
   */
  export type DataSyncFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * Filter, which DataSync to fetch.
     */
    where: DataSyncWhereUniqueInput
  }

  /**
   * DataSync findUniqueOrThrow
   */
  export type DataSyncFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * Filter, which DataSync to fetch.
     */
    where: DataSyncWhereUniqueInput
  }

  /**
   * DataSync findFirst
   */
  export type DataSyncFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * Filter, which DataSync to fetch.
     */
    where?: DataSyncWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataSyncs to fetch.
     */
    orderBy?: DataSyncOrderByWithRelationInput | DataSyncOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for DataSyncs.
     */
    cursor?: DataSyncWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataSyncs from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataSyncs.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of DataSyncs.
     */
    distinct?: DataSyncScalarFieldEnum | DataSyncScalarFieldEnum[]
  }

  /**
   * DataSync findFirstOrThrow
   */
  export type DataSyncFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * Filter, which DataSync to fetch.
     */
    where?: DataSyncWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataSyncs to fetch.
     */
    orderBy?: DataSyncOrderByWithRelationInput | DataSyncOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for DataSyncs.
     */
    cursor?: DataSyncWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataSyncs from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataSyncs.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of DataSyncs.
     */
    distinct?: DataSyncScalarFieldEnum | DataSyncScalarFieldEnum[]
  }

  /**
   * DataSync findMany
   */
  export type DataSyncFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * Filter, which DataSyncs to fetch.
     */
    where?: DataSyncWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataSyncs to fetch.
     */
    orderBy?: DataSyncOrderByWithRelationInput | DataSyncOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing DataSyncs.
     */
    cursor?: DataSyncWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataSyncs from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataSyncs.
     */
    skip?: number
    distinct?: DataSyncScalarFieldEnum | DataSyncScalarFieldEnum[]
  }

  /**
   * DataSync create
   */
  export type DataSyncCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * The data needed to create a DataSync.
     */
    data: XOR<DataSyncCreateInput, DataSyncUncheckedCreateInput>
  }

  /**
   * DataSync createMany
   */
  export type DataSyncCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many DataSyncs.
     */
    data: DataSyncCreateManyInput | DataSyncCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * DataSync createManyAndReturn
   */
  export type DataSyncCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * The data used to create many DataSyncs.
     */
    data: DataSyncCreateManyInput | DataSyncCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * DataSync update
   */
  export type DataSyncUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * The data needed to update a DataSync.
     */
    data: XOR<DataSyncUpdateInput, DataSyncUncheckedUpdateInput>
    /**
     * Choose, which DataSync to update.
     */
    where: DataSyncWhereUniqueInput
  }

  /**
   * DataSync updateMany
   */
  export type DataSyncUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update DataSyncs.
     */
    data: XOR<DataSyncUpdateManyMutationInput, DataSyncUncheckedUpdateManyInput>
    /**
     * Filter which DataSyncs to update
     */
    where?: DataSyncWhereInput
    /**
     * Limit how many DataSyncs to update.
     */
    limit?: number
  }

  /**
   * DataSync updateManyAndReturn
   */
  export type DataSyncUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * The data used to update DataSyncs.
     */
    data: XOR<DataSyncUpdateManyMutationInput, DataSyncUncheckedUpdateManyInput>
    /**
     * Filter which DataSyncs to update
     */
    where?: DataSyncWhereInput
    /**
     * Limit how many DataSyncs to update.
     */
    limit?: number
  }

  /**
   * DataSync upsert
   */
  export type DataSyncUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * The filter to search for the DataSync to update in case it exists.
     */
    where: DataSyncWhereUniqueInput
    /**
     * In case the DataSync found by the `where` argument doesn't exist, create a new DataSync with this data.
     */
    create: XOR<DataSyncCreateInput, DataSyncUncheckedCreateInput>
    /**
     * In case the DataSync was found with the provided `where` argument, update it with this data.
     */
    update: XOR<DataSyncUpdateInput, DataSyncUncheckedUpdateInput>
  }

  /**
   * DataSync delete
   */
  export type DataSyncDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
    /**
     * Filter which DataSync to delete.
     */
    where: DataSyncWhereUniqueInput
  }

  /**
   * DataSync deleteMany
   */
  export type DataSyncDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which DataSyncs to delete
     */
    where?: DataSyncWhereInput
    /**
     * Limit how many DataSyncs to delete.
     */
    limit?: number
  }

  /**
   * DataSync without action
   */
  export type DataSyncDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSync
     */
    select?: DataSyncSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSync
     */
    omit?: DataSyncOmit<ExtArgs> | null
  }


  /**
   * Enums
   */

  export const TransactionIsolationLevel: {
    ReadUncommitted: 'ReadUncommitted',
    ReadCommitted: 'ReadCommitted',
    RepeatableRead: 'RepeatableRead',
    Serializable: 'Serializable'
  };

  export type TransactionIsolationLevel = (typeof TransactionIsolationLevel)[keyof typeof TransactionIsolationLevel]


  export const QuestionScalarFieldEnum: {
    id: 'id',
    text: 'text',
    originalText: 'originalText',
    embedding: 'embedding',
    language: 'language',
    country: 'country',
    state: 'state',
    userId: 'userId',
    createdAt: 'createdAt',
    updatedAt: 'updatedAt'
  };

  export type QuestionScalarFieldEnum = (typeof QuestionScalarFieldEnum)[keyof typeof QuestionScalarFieldEnum]


  export const TopicScalarFieldEnum: {
    id: 'id',
    name: 'name',
    description: 'description',
    subtopic: 'subtopic',
    isSystemTopic: 'isSystemTopic',
    isActive: 'isActive',
    embedding: 'embedding',
    createdAt: 'createdAt',
    updatedAt: 'updatedAt'
  };

  export type TopicScalarFieldEnum = (typeof TopicScalarFieldEnum)[keyof typeof TopicScalarFieldEnum]


  export const TopicAssignmentScalarFieldEnum: {
    id: 'id',
    questionId: 'questionId',
    topicId: 'topicId',
    similarityScore: 'similarityScore',
    assignmentType: 'assignmentType',
    confidence: 'confidence',
    analysisRunId: 'analysisRunId',
    createdAt: 'createdAt'
  };

  export type TopicAssignmentScalarFieldEnum = (typeof TopicAssignmentScalarFieldEnum)[keyof typeof TopicAssignmentScalarFieldEnum]


  export const ClusterResultScalarFieldEnum: {
    id: 'id',
    questionId: 'questionId',
    clusterId: 'clusterId',
    clusterName: 'clusterName',
    analysisRunId: 'analysisRunId',
    createdAt: 'createdAt'
  };

  export type ClusterResultScalarFieldEnum = (typeof ClusterResultScalarFieldEnum)[keyof typeof ClusterResultScalarFieldEnum]


  export const AnalysisRunScalarFieldEnum: {
    id: 'id',
    status: 'status',
    mode: 'mode',
    sampleSize: 'sampleSize',
    similarityThreshold: 'similarityThreshold',
    embeddingModel: 'embeddingModel',
    gptModel: 'gptModel',
    totalQuestions: 'totalQuestions',
    matchedQuestions: 'matchedQuestions',
    newTopicsFound: 'newTopicsFound',
    processingTimeMs: 'processingTimeMs',
    errorMessage: 'errorMessage',
    startedAt: 'startedAt',
    completedAt: 'completedAt',
    createdBy: 'createdBy',
    config: 'config'
  };

  export type AnalysisRunScalarFieldEnum = (typeof AnalysisRunScalarFieldEnum)[keyof typeof AnalysisRunScalarFieldEnum]


  export const SystemConfigScalarFieldEnum: {
    id: 'id',
    key: 'key',
    value: 'value',
    type: 'type',
    updatedAt: 'updatedAt',
    updatedBy: 'updatedBy'
  };

  export type SystemConfigScalarFieldEnum = (typeof SystemConfigScalarFieldEnum)[keyof typeof SystemConfigScalarFieldEnum]


  export const UserSessionScalarFieldEnum: {
    id: 'id',
    sessionId: 'sessionId',
    userId: 'userId',
    isDevSession: 'isDevSession',
    ipAddress: 'ipAddress',
    userAgent: 'userAgent',
    country: 'country',
    region: 'region',
    createdAt: 'createdAt',
    lastActivity: 'lastActivity'
  };

  export type UserSessionScalarFieldEnum = (typeof UserSessionScalarFieldEnum)[keyof typeof UserSessionScalarFieldEnum]


  export const DataSyncScalarFieldEnum: {
    id: 'id',
    source: 'source',
    sourceUrl: 'sourceUrl',
    lastSyncAt: 'lastSyncAt',
    status: 'status',
    recordsProcessed: 'recordsProcessed',
    recordsAdded: 'recordsAdded',
    recordsUpdated: 'recordsUpdated',
    errorMessage: 'errorMessage'
  };

  export type DataSyncScalarFieldEnum = (typeof DataSyncScalarFieldEnum)[keyof typeof DataSyncScalarFieldEnum]


  export const SortOrder: {
    asc: 'asc',
    desc: 'desc'
  };

  export type SortOrder = (typeof SortOrder)[keyof typeof SortOrder]


  export const NullableJsonNullValueInput: {
    DbNull: typeof DbNull,
    JsonNull: typeof JsonNull
  };

  export type NullableJsonNullValueInput = (typeof NullableJsonNullValueInput)[keyof typeof NullableJsonNullValueInput]


  export const QueryMode: {
    default: 'default',
    insensitive: 'insensitive'
  };

  export type QueryMode = (typeof QueryMode)[keyof typeof QueryMode]


  export const NullsOrder: {
    first: 'first',
    last: 'last'
  };

  export type NullsOrder = (typeof NullsOrder)[keyof typeof NullsOrder]


  export const JsonNullValueFilter: {
    DbNull: typeof DbNull,
    JsonNull: typeof JsonNull,
    AnyNull: typeof AnyNull
  };

  export type JsonNullValueFilter = (typeof JsonNullValueFilter)[keyof typeof JsonNullValueFilter]


  /**
   * Field references
   */


  /**
   * Reference to a field of type 'String'
   */
  export type StringFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'String'>
    


  /**
   * Reference to a field of type 'String[]'
   */
  export type ListStringFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'String[]'>
    


  /**
   * Reference to a field of type 'Float[]'
   */
  export type ListFloatFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Float[]'>
    


  /**
   * Reference to a field of type 'Float'
   */
  export type FloatFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Float'>
    


  /**
   * Reference to a field of type 'DateTime'
   */
  export type DateTimeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'DateTime'>
    


  /**
   * Reference to a field of type 'DateTime[]'
   */
  export type ListDateTimeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'DateTime[]'>
    


  /**
   * Reference to a field of type 'Boolean'
   */
  export type BooleanFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Boolean'>
    


  /**
   * Reference to a field of type 'AssignmentType'
   */
  export type EnumAssignmentTypeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'AssignmentType'>
    


  /**
   * Reference to a field of type 'AssignmentType[]'
   */
  export type ListEnumAssignmentTypeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'AssignmentType[]'>
    


  /**
   * Reference to a field of type 'Int'
   */
  export type IntFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Int'>
    


  /**
   * Reference to a field of type 'Int[]'
   */
  export type ListIntFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Int[]'>
    


  /**
   * Reference to a field of type 'AnalysisStatus'
   */
  export type EnumAnalysisStatusFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'AnalysisStatus'>
    


  /**
   * Reference to a field of type 'AnalysisStatus[]'
   */
  export type ListEnumAnalysisStatusFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'AnalysisStatus[]'>
    


  /**
   * Reference to a field of type 'Json'
   */
  export type JsonFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Json'>
    


  /**
   * Reference to a field of type 'QueryMode'
   */
  export type EnumQueryModeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'QueryMode'>
    


  /**
   * Reference to a field of type 'ConfigType'
   */
  export type EnumConfigTypeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'ConfigType'>
    


  /**
   * Reference to a field of type 'ConfigType[]'
   */
  export type ListEnumConfigTypeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'ConfigType[]'>
    


  /**
   * Reference to a field of type 'SyncStatus'
   */
  export type EnumSyncStatusFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'SyncStatus'>
    


  /**
   * Reference to a field of type 'SyncStatus[]'
   */
  export type ListEnumSyncStatusFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'SyncStatus[]'>
    
  /**
   * Deep Input Types
   */


  export type QuestionWhereInput = {
    AND?: QuestionWhereInput | QuestionWhereInput[]
    OR?: QuestionWhereInput[]
    NOT?: QuestionWhereInput | QuestionWhereInput[]
    id?: StringFilter<"Question"> | string
    text?: StringFilter<"Question"> | string
    originalText?: StringNullableFilter<"Question"> | string | null
    embedding?: FloatNullableListFilter<"Question">
    language?: StringFilter<"Question"> | string
    country?: StringNullableFilter<"Question"> | string | null
    state?: StringNullableFilter<"Question"> | string | null
    userId?: StringNullableFilter<"Question"> | string | null
    createdAt?: DateTimeFilter<"Question"> | Date | string
    updatedAt?: DateTimeFilter<"Question"> | Date | string
    topicAssignments?: TopicAssignmentListRelationFilter
    clusterResults?: ClusterResultListRelationFilter
  }

  export type QuestionOrderByWithRelationInput = {
    id?: SortOrder
    text?: SortOrder
    originalText?: SortOrderInput | SortOrder
    embedding?: SortOrder
    language?: SortOrder
    country?: SortOrderInput | SortOrder
    state?: SortOrderInput | SortOrder
    userId?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    topicAssignments?: TopicAssignmentOrderByRelationAggregateInput
    clusterResults?: ClusterResultOrderByRelationAggregateInput
  }

  export type QuestionWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    AND?: QuestionWhereInput | QuestionWhereInput[]
    OR?: QuestionWhereInput[]
    NOT?: QuestionWhereInput | QuestionWhereInput[]
    text?: StringFilter<"Question"> | string
    originalText?: StringNullableFilter<"Question"> | string | null
    embedding?: FloatNullableListFilter<"Question">
    language?: StringFilter<"Question"> | string
    country?: StringNullableFilter<"Question"> | string | null
    state?: StringNullableFilter<"Question"> | string | null
    userId?: StringNullableFilter<"Question"> | string | null
    createdAt?: DateTimeFilter<"Question"> | Date | string
    updatedAt?: DateTimeFilter<"Question"> | Date | string
    topicAssignments?: TopicAssignmentListRelationFilter
    clusterResults?: ClusterResultListRelationFilter
  }, "id">

  export type QuestionOrderByWithAggregationInput = {
    id?: SortOrder
    text?: SortOrder
    originalText?: SortOrderInput | SortOrder
    embedding?: SortOrder
    language?: SortOrder
    country?: SortOrderInput | SortOrder
    state?: SortOrderInput | SortOrder
    userId?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    _count?: QuestionCountOrderByAggregateInput
    _avg?: QuestionAvgOrderByAggregateInput
    _max?: QuestionMaxOrderByAggregateInput
    _min?: QuestionMinOrderByAggregateInput
    _sum?: QuestionSumOrderByAggregateInput
  }

  export type QuestionScalarWhereWithAggregatesInput = {
    AND?: QuestionScalarWhereWithAggregatesInput | QuestionScalarWhereWithAggregatesInput[]
    OR?: QuestionScalarWhereWithAggregatesInput[]
    NOT?: QuestionScalarWhereWithAggregatesInput | QuestionScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"Question"> | string
    text?: StringWithAggregatesFilter<"Question"> | string
    originalText?: StringNullableWithAggregatesFilter<"Question"> | string | null
    embedding?: FloatNullableListFilter<"Question">
    language?: StringWithAggregatesFilter<"Question"> | string
    country?: StringNullableWithAggregatesFilter<"Question"> | string | null
    state?: StringNullableWithAggregatesFilter<"Question"> | string | null
    userId?: StringNullableWithAggregatesFilter<"Question"> | string | null
    createdAt?: DateTimeWithAggregatesFilter<"Question"> | Date | string
    updatedAt?: DateTimeWithAggregatesFilter<"Question"> | Date | string
  }

  export type TopicWhereInput = {
    AND?: TopicWhereInput | TopicWhereInput[]
    OR?: TopicWhereInput[]
    NOT?: TopicWhereInput | TopicWhereInput[]
    id?: StringFilter<"Topic"> | string
    name?: StringFilter<"Topic"> | string
    description?: StringNullableFilter<"Topic"> | string | null
    subtopic?: StringNullableFilter<"Topic"> | string | null
    isSystemTopic?: BoolFilter<"Topic"> | boolean
    isActive?: BoolFilter<"Topic"> | boolean
    embedding?: FloatNullableListFilter<"Topic">
    createdAt?: DateTimeFilter<"Topic"> | Date | string
    updatedAt?: DateTimeFilter<"Topic"> | Date | string
    topicAssignments?: TopicAssignmentListRelationFilter
  }

  export type TopicOrderByWithRelationInput = {
    id?: SortOrder
    name?: SortOrder
    description?: SortOrderInput | SortOrder
    subtopic?: SortOrderInput | SortOrder
    isSystemTopic?: SortOrder
    isActive?: SortOrder
    embedding?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    topicAssignments?: TopicAssignmentOrderByRelationAggregateInput
  }

  export type TopicWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    AND?: TopicWhereInput | TopicWhereInput[]
    OR?: TopicWhereInput[]
    NOT?: TopicWhereInput | TopicWhereInput[]
    name?: StringFilter<"Topic"> | string
    description?: StringNullableFilter<"Topic"> | string | null
    subtopic?: StringNullableFilter<"Topic"> | string | null
    isSystemTopic?: BoolFilter<"Topic"> | boolean
    isActive?: BoolFilter<"Topic"> | boolean
    embedding?: FloatNullableListFilter<"Topic">
    createdAt?: DateTimeFilter<"Topic"> | Date | string
    updatedAt?: DateTimeFilter<"Topic"> | Date | string
    topicAssignments?: TopicAssignmentListRelationFilter
  }, "id">

  export type TopicOrderByWithAggregationInput = {
    id?: SortOrder
    name?: SortOrder
    description?: SortOrderInput | SortOrder
    subtopic?: SortOrderInput | SortOrder
    isSystemTopic?: SortOrder
    isActive?: SortOrder
    embedding?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
    _count?: TopicCountOrderByAggregateInput
    _avg?: TopicAvgOrderByAggregateInput
    _max?: TopicMaxOrderByAggregateInput
    _min?: TopicMinOrderByAggregateInput
    _sum?: TopicSumOrderByAggregateInput
  }

  export type TopicScalarWhereWithAggregatesInput = {
    AND?: TopicScalarWhereWithAggregatesInput | TopicScalarWhereWithAggregatesInput[]
    OR?: TopicScalarWhereWithAggregatesInput[]
    NOT?: TopicScalarWhereWithAggregatesInput | TopicScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"Topic"> | string
    name?: StringWithAggregatesFilter<"Topic"> | string
    description?: StringNullableWithAggregatesFilter<"Topic"> | string | null
    subtopic?: StringNullableWithAggregatesFilter<"Topic"> | string | null
    isSystemTopic?: BoolWithAggregatesFilter<"Topic"> | boolean
    isActive?: BoolWithAggregatesFilter<"Topic"> | boolean
    embedding?: FloatNullableListFilter<"Topic">
    createdAt?: DateTimeWithAggregatesFilter<"Topic"> | Date | string
    updatedAt?: DateTimeWithAggregatesFilter<"Topic"> | Date | string
  }

  export type TopicAssignmentWhereInput = {
    AND?: TopicAssignmentWhereInput | TopicAssignmentWhereInput[]
    OR?: TopicAssignmentWhereInput[]
    NOT?: TopicAssignmentWhereInput | TopicAssignmentWhereInput[]
    id?: StringFilter<"TopicAssignment"> | string
    questionId?: StringFilter<"TopicAssignment"> | string
    topicId?: StringFilter<"TopicAssignment"> | string
    similarityScore?: FloatNullableFilter<"TopicAssignment"> | number | null
    assignmentType?: EnumAssignmentTypeFilter<"TopicAssignment"> | $Enums.AssignmentType
    confidence?: FloatNullableFilter<"TopicAssignment"> | number | null
    analysisRunId?: StringFilter<"TopicAssignment"> | string
    createdAt?: DateTimeFilter<"TopicAssignment"> | Date | string
    question?: XOR<QuestionScalarRelationFilter, QuestionWhereInput>
    topic?: XOR<TopicScalarRelationFilter, TopicWhereInput>
    analysisRun?: XOR<AnalysisRunScalarRelationFilter, AnalysisRunWhereInput>
  }

  export type TopicAssignmentOrderByWithRelationInput = {
    id?: SortOrder
    questionId?: SortOrder
    topicId?: SortOrder
    similarityScore?: SortOrderInput | SortOrder
    assignmentType?: SortOrder
    confidence?: SortOrderInput | SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
    question?: QuestionOrderByWithRelationInput
    topic?: TopicOrderByWithRelationInput
    analysisRun?: AnalysisRunOrderByWithRelationInput
  }

  export type TopicAssignmentWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    questionId_topicId_analysisRunId?: TopicAssignmentQuestionIdTopicIdAnalysisRunIdCompoundUniqueInput
    AND?: TopicAssignmentWhereInput | TopicAssignmentWhereInput[]
    OR?: TopicAssignmentWhereInput[]
    NOT?: TopicAssignmentWhereInput | TopicAssignmentWhereInput[]
    questionId?: StringFilter<"TopicAssignment"> | string
    topicId?: StringFilter<"TopicAssignment"> | string
    similarityScore?: FloatNullableFilter<"TopicAssignment"> | number | null
    assignmentType?: EnumAssignmentTypeFilter<"TopicAssignment"> | $Enums.AssignmentType
    confidence?: FloatNullableFilter<"TopicAssignment"> | number | null
    analysisRunId?: StringFilter<"TopicAssignment"> | string
    createdAt?: DateTimeFilter<"TopicAssignment"> | Date | string
    question?: XOR<QuestionScalarRelationFilter, QuestionWhereInput>
    topic?: XOR<TopicScalarRelationFilter, TopicWhereInput>
    analysisRun?: XOR<AnalysisRunScalarRelationFilter, AnalysisRunWhereInput>
  }, "id" | "questionId_topicId_analysisRunId">

  export type TopicAssignmentOrderByWithAggregationInput = {
    id?: SortOrder
    questionId?: SortOrder
    topicId?: SortOrder
    similarityScore?: SortOrderInput | SortOrder
    assignmentType?: SortOrder
    confidence?: SortOrderInput | SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
    _count?: TopicAssignmentCountOrderByAggregateInput
    _avg?: TopicAssignmentAvgOrderByAggregateInput
    _max?: TopicAssignmentMaxOrderByAggregateInput
    _min?: TopicAssignmentMinOrderByAggregateInput
    _sum?: TopicAssignmentSumOrderByAggregateInput
  }

  export type TopicAssignmentScalarWhereWithAggregatesInput = {
    AND?: TopicAssignmentScalarWhereWithAggregatesInput | TopicAssignmentScalarWhereWithAggregatesInput[]
    OR?: TopicAssignmentScalarWhereWithAggregatesInput[]
    NOT?: TopicAssignmentScalarWhereWithAggregatesInput | TopicAssignmentScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"TopicAssignment"> | string
    questionId?: StringWithAggregatesFilter<"TopicAssignment"> | string
    topicId?: StringWithAggregatesFilter<"TopicAssignment"> | string
    similarityScore?: FloatNullableWithAggregatesFilter<"TopicAssignment"> | number | null
    assignmentType?: EnumAssignmentTypeWithAggregatesFilter<"TopicAssignment"> | $Enums.AssignmentType
    confidence?: FloatNullableWithAggregatesFilter<"TopicAssignment"> | number | null
    analysisRunId?: StringWithAggregatesFilter<"TopicAssignment"> | string
    createdAt?: DateTimeWithAggregatesFilter<"TopicAssignment"> | Date | string
  }

  export type ClusterResultWhereInput = {
    AND?: ClusterResultWhereInput | ClusterResultWhereInput[]
    OR?: ClusterResultWhereInput[]
    NOT?: ClusterResultWhereInput | ClusterResultWhereInput[]
    id?: StringFilter<"ClusterResult"> | string
    questionId?: StringFilter<"ClusterResult"> | string
    clusterId?: IntFilter<"ClusterResult"> | number
    clusterName?: StringNullableFilter<"ClusterResult"> | string | null
    analysisRunId?: StringFilter<"ClusterResult"> | string
    createdAt?: DateTimeFilter<"ClusterResult"> | Date | string
    question?: XOR<QuestionScalarRelationFilter, QuestionWhereInput>
    analysisRun?: XOR<AnalysisRunScalarRelationFilter, AnalysisRunWhereInput>
  }

  export type ClusterResultOrderByWithRelationInput = {
    id?: SortOrder
    questionId?: SortOrder
    clusterId?: SortOrder
    clusterName?: SortOrderInput | SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
    question?: QuestionOrderByWithRelationInput
    analysisRun?: AnalysisRunOrderByWithRelationInput
  }

  export type ClusterResultWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    questionId_analysisRunId?: ClusterResultQuestionIdAnalysisRunIdCompoundUniqueInput
    AND?: ClusterResultWhereInput | ClusterResultWhereInput[]
    OR?: ClusterResultWhereInput[]
    NOT?: ClusterResultWhereInput | ClusterResultWhereInput[]
    questionId?: StringFilter<"ClusterResult"> | string
    clusterId?: IntFilter<"ClusterResult"> | number
    clusterName?: StringNullableFilter<"ClusterResult"> | string | null
    analysisRunId?: StringFilter<"ClusterResult"> | string
    createdAt?: DateTimeFilter<"ClusterResult"> | Date | string
    question?: XOR<QuestionScalarRelationFilter, QuestionWhereInput>
    analysisRun?: XOR<AnalysisRunScalarRelationFilter, AnalysisRunWhereInput>
  }, "id" | "questionId_analysisRunId">

  export type ClusterResultOrderByWithAggregationInput = {
    id?: SortOrder
    questionId?: SortOrder
    clusterId?: SortOrder
    clusterName?: SortOrderInput | SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
    _count?: ClusterResultCountOrderByAggregateInput
    _avg?: ClusterResultAvgOrderByAggregateInput
    _max?: ClusterResultMaxOrderByAggregateInput
    _min?: ClusterResultMinOrderByAggregateInput
    _sum?: ClusterResultSumOrderByAggregateInput
  }

  export type ClusterResultScalarWhereWithAggregatesInput = {
    AND?: ClusterResultScalarWhereWithAggregatesInput | ClusterResultScalarWhereWithAggregatesInput[]
    OR?: ClusterResultScalarWhereWithAggregatesInput[]
    NOT?: ClusterResultScalarWhereWithAggregatesInput | ClusterResultScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"ClusterResult"> | string
    questionId?: StringWithAggregatesFilter<"ClusterResult"> | string
    clusterId?: IntWithAggregatesFilter<"ClusterResult"> | number
    clusterName?: StringNullableWithAggregatesFilter<"ClusterResult"> | string | null
    analysisRunId?: StringWithAggregatesFilter<"ClusterResult"> | string
    createdAt?: DateTimeWithAggregatesFilter<"ClusterResult"> | Date | string
  }

  export type AnalysisRunWhereInput = {
    AND?: AnalysisRunWhereInput | AnalysisRunWhereInput[]
    OR?: AnalysisRunWhereInput[]
    NOT?: AnalysisRunWhereInput | AnalysisRunWhereInput[]
    id?: StringFilter<"AnalysisRun"> | string
    status?: EnumAnalysisStatusFilter<"AnalysisRun"> | $Enums.AnalysisStatus
    mode?: StringFilter<"AnalysisRun"> | string
    sampleSize?: IntNullableFilter<"AnalysisRun"> | number | null
    similarityThreshold?: FloatFilter<"AnalysisRun"> | number
    embeddingModel?: StringFilter<"AnalysisRun"> | string
    gptModel?: StringFilter<"AnalysisRun"> | string
    totalQuestions?: IntNullableFilter<"AnalysisRun"> | number | null
    matchedQuestions?: IntNullableFilter<"AnalysisRun"> | number | null
    newTopicsFound?: IntNullableFilter<"AnalysisRun"> | number | null
    processingTimeMs?: IntNullableFilter<"AnalysisRun"> | number | null
    errorMessage?: StringNullableFilter<"AnalysisRun"> | string | null
    startedAt?: DateTimeFilter<"AnalysisRun"> | Date | string
    completedAt?: DateTimeNullableFilter<"AnalysisRun"> | Date | string | null
    createdBy?: StringNullableFilter<"AnalysisRun"> | string | null
    config?: JsonNullableFilter<"AnalysisRun">
    topicAssignments?: TopicAssignmentListRelationFilter
    clusterResults?: ClusterResultListRelationFilter
  }

  export type AnalysisRunOrderByWithRelationInput = {
    id?: SortOrder
    status?: SortOrder
    mode?: SortOrder
    sampleSize?: SortOrderInput | SortOrder
    similarityThreshold?: SortOrder
    embeddingModel?: SortOrder
    gptModel?: SortOrder
    totalQuestions?: SortOrderInput | SortOrder
    matchedQuestions?: SortOrderInput | SortOrder
    newTopicsFound?: SortOrderInput | SortOrder
    processingTimeMs?: SortOrderInput | SortOrder
    errorMessage?: SortOrderInput | SortOrder
    startedAt?: SortOrder
    completedAt?: SortOrderInput | SortOrder
    createdBy?: SortOrderInput | SortOrder
    config?: SortOrderInput | SortOrder
    topicAssignments?: TopicAssignmentOrderByRelationAggregateInput
    clusterResults?: ClusterResultOrderByRelationAggregateInput
  }

  export type AnalysisRunWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    AND?: AnalysisRunWhereInput | AnalysisRunWhereInput[]
    OR?: AnalysisRunWhereInput[]
    NOT?: AnalysisRunWhereInput | AnalysisRunWhereInput[]
    status?: EnumAnalysisStatusFilter<"AnalysisRun"> | $Enums.AnalysisStatus
    mode?: StringFilter<"AnalysisRun"> | string
    sampleSize?: IntNullableFilter<"AnalysisRun"> | number | null
    similarityThreshold?: FloatFilter<"AnalysisRun"> | number
    embeddingModel?: StringFilter<"AnalysisRun"> | string
    gptModel?: StringFilter<"AnalysisRun"> | string
    totalQuestions?: IntNullableFilter<"AnalysisRun"> | number | null
    matchedQuestions?: IntNullableFilter<"AnalysisRun"> | number | null
    newTopicsFound?: IntNullableFilter<"AnalysisRun"> | number | null
    processingTimeMs?: IntNullableFilter<"AnalysisRun"> | number | null
    errorMessage?: StringNullableFilter<"AnalysisRun"> | string | null
    startedAt?: DateTimeFilter<"AnalysisRun"> | Date | string
    completedAt?: DateTimeNullableFilter<"AnalysisRun"> | Date | string | null
    createdBy?: StringNullableFilter<"AnalysisRun"> | string | null
    config?: JsonNullableFilter<"AnalysisRun">
    topicAssignments?: TopicAssignmentListRelationFilter
    clusterResults?: ClusterResultListRelationFilter
  }, "id">

  export type AnalysisRunOrderByWithAggregationInput = {
    id?: SortOrder
    status?: SortOrder
    mode?: SortOrder
    sampleSize?: SortOrderInput | SortOrder
    similarityThreshold?: SortOrder
    embeddingModel?: SortOrder
    gptModel?: SortOrder
    totalQuestions?: SortOrderInput | SortOrder
    matchedQuestions?: SortOrderInput | SortOrder
    newTopicsFound?: SortOrderInput | SortOrder
    processingTimeMs?: SortOrderInput | SortOrder
    errorMessage?: SortOrderInput | SortOrder
    startedAt?: SortOrder
    completedAt?: SortOrderInput | SortOrder
    createdBy?: SortOrderInput | SortOrder
    config?: SortOrderInput | SortOrder
    _count?: AnalysisRunCountOrderByAggregateInput
    _avg?: AnalysisRunAvgOrderByAggregateInput
    _max?: AnalysisRunMaxOrderByAggregateInput
    _min?: AnalysisRunMinOrderByAggregateInput
    _sum?: AnalysisRunSumOrderByAggregateInput
  }

  export type AnalysisRunScalarWhereWithAggregatesInput = {
    AND?: AnalysisRunScalarWhereWithAggregatesInput | AnalysisRunScalarWhereWithAggregatesInput[]
    OR?: AnalysisRunScalarWhereWithAggregatesInput[]
    NOT?: AnalysisRunScalarWhereWithAggregatesInput | AnalysisRunScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"AnalysisRun"> | string
    status?: EnumAnalysisStatusWithAggregatesFilter<"AnalysisRun"> | $Enums.AnalysisStatus
    mode?: StringWithAggregatesFilter<"AnalysisRun"> | string
    sampleSize?: IntNullableWithAggregatesFilter<"AnalysisRun"> | number | null
    similarityThreshold?: FloatWithAggregatesFilter<"AnalysisRun"> | number
    embeddingModel?: StringWithAggregatesFilter<"AnalysisRun"> | string
    gptModel?: StringWithAggregatesFilter<"AnalysisRun"> | string
    totalQuestions?: IntNullableWithAggregatesFilter<"AnalysisRun"> | number | null
    matchedQuestions?: IntNullableWithAggregatesFilter<"AnalysisRun"> | number | null
    newTopicsFound?: IntNullableWithAggregatesFilter<"AnalysisRun"> | number | null
    processingTimeMs?: IntNullableWithAggregatesFilter<"AnalysisRun"> | number | null
    errorMessage?: StringNullableWithAggregatesFilter<"AnalysisRun"> | string | null
    startedAt?: DateTimeWithAggregatesFilter<"AnalysisRun"> | Date | string
    completedAt?: DateTimeNullableWithAggregatesFilter<"AnalysisRun"> | Date | string | null
    createdBy?: StringNullableWithAggregatesFilter<"AnalysisRun"> | string | null
    config?: JsonNullableWithAggregatesFilter<"AnalysisRun">
  }

  export type SystemConfigWhereInput = {
    AND?: SystemConfigWhereInput | SystemConfigWhereInput[]
    OR?: SystemConfigWhereInput[]
    NOT?: SystemConfigWhereInput | SystemConfigWhereInput[]
    id?: StringFilter<"SystemConfig"> | string
    key?: StringFilter<"SystemConfig"> | string
    value?: StringFilter<"SystemConfig"> | string
    type?: EnumConfigTypeFilter<"SystemConfig"> | $Enums.ConfigType
    updatedAt?: DateTimeFilter<"SystemConfig"> | Date | string
    updatedBy?: StringNullableFilter<"SystemConfig"> | string | null
  }

  export type SystemConfigOrderByWithRelationInput = {
    id?: SortOrder
    key?: SortOrder
    value?: SortOrder
    type?: SortOrder
    updatedAt?: SortOrder
    updatedBy?: SortOrderInput | SortOrder
  }

  export type SystemConfigWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    key?: string
    AND?: SystemConfigWhereInput | SystemConfigWhereInput[]
    OR?: SystemConfigWhereInput[]
    NOT?: SystemConfigWhereInput | SystemConfigWhereInput[]
    value?: StringFilter<"SystemConfig"> | string
    type?: EnumConfigTypeFilter<"SystemConfig"> | $Enums.ConfigType
    updatedAt?: DateTimeFilter<"SystemConfig"> | Date | string
    updatedBy?: StringNullableFilter<"SystemConfig"> | string | null
  }, "id" | "key">

  export type SystemConfigOrderByWithAggregationInput = {
    id?: SortOrder
    key?: SortOrder
    value?: SortOrder
    type?: SortOrder
    updatedAt?: SortOrder
    updatedBy?: SortOrderInput | SortOrder
    _count?: SystemConfigCountOrderByAggregateInput
    _max?: SystemConfigMaxOrderByAggregateInput
    _min?: SystemConfigMinOrderByAggregateInput
  }

  export type SystemConfigScalarWhereWithAggregatesInput = {
    AND?: SystemConfigScalarWhereWithAggregatesInput | SystemConfigScalarWhereWithAggregatesInput[]
    OR?: SystemConfigScalarWhereWithAggregatesInput[]
    NOT?: SystemConfigScalarWhereWithAggregatesInput | SystemConfigScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"SystemConfig"> | string
    key?: StringWithAggregatesFilter<"SystemConfig"> | string
    value?: StringWithAggregatesFilter<"SystemConfig"> | string
    type?: EnumConfigTypeWithAggregatesFilter<"SystemConfig"> | $Enums.ConfigType
    updatedAt?: DateTimeWithAggregatesFilter<"SystemConfig"> | Date | string
    updatedBy?: StringNullableWithAggregatesFilter<"SystemConfig"> | string | null
  }

  export type UserSessionWhereInput = {
    AND?: UserSessionWhereInput | UserSessionWhereInput[]
    OR?: UserSessionWhereInput[]
    NOT?: UserSessionWhereInput | UserSessionWhereInput[]
    id?: StringFilter<"UserSession"> | string
    sessionId?: StringFilter<"UserSession"> | string
    userId?: StringNullableFilter<"UserSession"> | string | null
    isDevSession?: BoolFilter<"UserSession"> | boolean
    ipAddress?: StringNullableFilter<"UserSession"> | string | null
    userAgent?: StringNullableFilter<"UserSession"> | string | null
    country?: StringNullableFilter<"UserSession"> | string | null
    region?: StringNullableFilter<"UserSession"> | string | null
    createdAt?: DateTimeFilter<"UserSession"> | Date | string
    lastActivity?: DateTimeFilter<"UserSession"> | Date | string
  }

  export type UserSessionOrderByWithRelationInput = {
    id?: SortOrder
    sessionId?: SortOrder
    userId?: SortOrderInput | SortOrder
    isDevSession?: SortOrder
    ipAddress?: SortOrderInput | SortOrder
    userAgent?: SortOrderInput | SortOrder
    country?: SortOrderInput | SortOrder
    region?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    lastActivity?: SortOrder
  }

  export type UserSessionWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    sessionId?: string
    AND?: UserSessionWhereInput | UserSessionWhereInput[]
    OR?: UserSessionWhereInput[]
    NOT?: UserSessionWhereInput | UserSessionWhereInput[]
    userId?: StringNullableFilter<"UserSession"> | string | null
    isDevSession?: BoolFilter<"UserSession"> | boolean
    ipAddress?: StringNullableFilter<"UserSession"> | string | null
    userAgent?: StringNullableFilter<"UserSession"> | string | null
    country?: StringNullableFilter<"UserSession"> | string | null
    region?: StringNullableFilter<"UserSession"> | string | null
    createdAt?: DateTimeFilter<"UserSession"> | Date | string
    lastActivity?: DateTimeFilter<"UserSession"> | Date | string
  }, "id" | "sessionId">

  export type UserSessionOrderByWithAggregationInput = {
    id?: SortOrder
    sessionId?: SortOrder
    userId?: SortOrderInput | SortOrder
    isDevSession?: SortOrder
    ipAddress?: SortOrderInput | SortOrder
    userAgent?: SortOrderInput | SortOrder
    country?: SortOrderInput | SortOrder
    region?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    lastActivity?: SortOrder
    _count?: UserSessionCountOrderByAggregateInput
    _max?: UserSessionMaxOrderByAggregateInput
    _min?: UserSessionMinOrderByAggregateInput
  }

  export type UserSessionScalarWhereWithAggregatesInput = {
    AND?: UserSessionScalarWhereWithAggregatesInput | UserSessionScalarWhereWithAggregatesInput[]
    OR?: UserSessionScalarWhereWithAggregatesInput[]
    NOT?: UserSessionScalarWhereWithAggregatesInput | UserSessionScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"UserSession"> | string
    sessionId?: StringWithAggregatesFilter<"UserSession"> | string
    userId?: StringNullableWithAggregatesFilter<"UserSession"> | string | null
    isDevSession?: BoolWithAggregatesFilter<"UserSession"> | boolean
    ipAddress?: StringNullableWithAggregatesFilter<"UserSession"> | string | null
    userAgent?: StringNullableWithAggregatesFilter<"UserSession"> | string | null
    country?: StringNullableWithAggregatesFilter<"UserSession"> | string | null
    region?: StringNullableWithAggregatesFilter<"UserSession"> | string | null
    createdAt?: DateTimeWithAggregatesFilter<"UserSession"> | Date | string
    lastActivity?: DateTimeWithAggregatesFilter<"UserSession"> | Date | string
  }

  export type DataSyncWhereInput = {
    AND?: DataSyncWhereInput | DataSyncWhereInput[]
    OR?: DataSyncWhereInput[]
    NOT?: DataSyncWhereInput | DataSyncWhereInput[]
    id?: StringFilter<"DataSync"> | string
    source?: StringFilter<"DataSync"> | string
    sourceUrl?: StringNullableFilter<"DataSync"> | string | null
    lastSyncAt?: DateTimeFilter<"DataSync"> | Date | string
    status?: EnumSyncStatusFilter<"DataSync"> | $Enums.SyncStatus
    recordsProcessed?: IntNullableFilter<"DataSync"> | number | null
    recordsAdded?: IntNullableFilter<"DataSync"> | number | null
    recordsUpdated?: IntNullableFilter<"DataSync"> | number | null
    errorMessage?: StringNullableFilter<"DataSync"> | string | null
  }

  export type DataSyncOrderByWithRelationInput = {
    id?: SortOrder
    source?: SortOrder
    sourceUrl?: SortOrderInput | SortOrder
    lastSyncAt?: SortOrder
    status?: SortOrder
    recordsProcessed?: SortOrderInput | SortOrder
    recordsAdded?: SortOrderInput | SortOrder
    recordsUpdated?: SortOrderInput | SortOrder
    errorMessage?: SortOrderInput | SortOrder
  }

  export type DataSyncWhereUniqueInput = Prisma.AtLeast<{
    id?: string
    AND?: DataSyncWhereInput | DataSyncWhereInput[]
    OR?: DataSyncWhereInput[]
    NOT?: DataSyncWhereInput | DataSyncWhereInput[]
    source?: StringFilter<"DataSync"> | string
    sourceUrl?: StringNullableFilter<"DataSync"> | string | null
    lastSyncAt?: DateTimeFilter<"DataSync"> | Date | string
    status?: EnumSyncStatusFilter<"DataSync"> | $Enums.SyncStatus
    recordsProcessed?: IntNullableFilter<"DataSync"> | number | null
    recordsAdded?: IntNullableFilter<"DataSync"> | number | null
    recordsUpdated?: IntNullableFilter<"DataSync"> | number | null
    errorMessage?: StringNullableFilter<"DataSync"> | string | null
  }, "id">

  export type DataSyncOrderByWithAggregationInput = {
    id?: SortOrder
    source?: SortOrder
    sourceUrl?: SortOrderInput | SortOrder
    lastSyncAt?: SortOrder
    status?: SortOrder
    recordsProcessed?: SortOrderInput | SortOrder
    recordsAdded?: SortOrderInput | SortOrder
    recordsUpdated?: SortOrderInput | SortOrder
    errorMessage?: SortOrderInput | SortOrder
    _count?: DataSyncCountOrderByAggregateInput
    _avg?: DataSyncAvgOrderByAggregateInput
    _max?: DataSyncMaxOrderByAggregateInput
    _min?: DataSyncMinOrderByAggregateInput
    _sum?: DataSyncSumOrderByAggregateInput
  }

  export type DataSyncScalarWhereWithAggregatesInput = {
    AND?: DataSyncScalarWhereWithAggregatesInput | DataSyncScalarWhereWithAggregatesInput[]
    OR?: DataSyncScalarWhereWithAggregatesInput[]
    NOT?: DataSyncScalarWhereWithAggregatesInput | DataSyncScalarWhereWithAggregatesInput[]
    id?: StringWithAggregatesFilter<"DataSync"> | string
    source?: StringWithAggregatesFilter<"DataSync"> | string
    sourceUrl?: StringNullableWithAggregatesFilter<"DataSync"> | string | null
    lastSyncAt?: DateTimeWithAggregatesFilter<"DataSync"> | Date | string
    status?: EnumSyncStatusWithAggregatesFilter<"DataSync"> | $Enums.SyncStatus
    recordsProcessed?: IntNullableWithAggregatesFilter<"DataSync"> | number | null
    recordsAdded?: IntNullableWithAggregatesFilter<"DataSync"> | number | null
    recordsUpdated?: IntNullableWithAggregatesFilter<"DataSync"> | number | null
    errorMessage?: StringNullableWithAggregatesFilter<"DataSync"> | string | null
  }

  export type QuestionCreateInput = {
    id?: string
    text: string
    originalText?: string | null
    embedding?: QuestionCreateembeddingInput | number[]
    language?: string
    country?: string | null
    state?: string | null
    userId?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
    topicAssignments?: TopicAssignmentCreateNestedManyWithoutQuestionInput
    clusterResults?: ClusterResultCreateNestedManyWithoutQuestionInput
  }

  export type QuestionUncheckedCreateInput = {
    id?: string
    text: string
    originalText?: string | null
    embedding?: QuestionCreateembeddingInput | number[]
    language?: string
    country?: string | null
    state?: string | null
    userId?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
    topicAssignments?: TopicAssignmentUncheckedCreateNestedManyWithoutQuestionInput
    clusterResults?: ClusterResultUncheckedCreateNestedManyWithoutQuestionInput
  }

  export type QuestionUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    text?: StringFieldUpdateOperationsInput | string
    originalText?: NullableStringFieldUpdateOperationsInput | string | null
    embedding?: QuestionUpdateembeddingInput | number[]
    language?: StringFieldUpdateOperationsInput | string
    country?: NullableStringFieldUpdateOperationsInput | string | null
    state?: NullableStringFieldUpdateOperationsInput | string | null
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    topicAssignments?: TopicAssignmentUpdateManyWithoutQuestionNestedInput
    clusterResults?: ClusterResultUpdateManyWithoutQuestionNestedInput
  }

  export type QuestionUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    text?: StringFieldUpdateOperationsInput | string
    originalText?: NullableStringFieldUpdateOperationsInput | string | null
    embedding?: QuestionUpdateembeddingInput | number[]
    language?: StringFieldUpdateOperationsInput | string
    country?: NullableStringFieldUpdateOperationsInput | string | null
    state?: NullableStringFieldUpdateOperationsInput | string | null
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    topicAssignments?: TopicAssignmentUncheckedUpdateManyWithoutQuestionNestedInput
    clusterResults?: ClusterResultUncheckedUpdateManyWithoutQuestionNestedInput
  }

  export type QuestionCreateManyInput = {
    id?: string
    text: string
    originalText?: string | null
    embedding?: QuestionCreateembeddingInput | number[]
    language?: string
    country?: string | null
    state?: string | null
    userId?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type QuestionUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    text?: StringFieldUpdateOperationsInput | string
    originalText?: NullableStringFieldUpdateOperationsInput | string | null
    embedding?: QuestionUpdateembeddingInput | number[]
    language?: StringFieldUpdateOperationsInput | string
    country?: NullableStringFieldUpdateOperationsInput | string | null
    state?: NullableStringFieldUpdateOperationsInput | string | null
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type QuestionUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    text?: StringFieldUpdateOperationsInput | string
    originalText?: NullableStringFieldUpdateOperationsInput | string | null
    embedding?: QuestionUpdateembeddingInput | number[]
    language?: StringFieldUpdateOperationsInput | string
    country?: NullableStringFieldUpdateOperationsInput | string | null
    state?: NullableStringFieldUpdateOperationsInput | string | null
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicCreateInput = {
    id?: string
    name: string
    description?: string | null
    subtopic?: string | null
    isSystemTopic?: boolean
    isActive?: boolean
    embedding?: TopicCreateembeddingInput | number[]
    createdAt?: Date | string
    updatedAt?: Date | string
    topicAssignments?: TopicAssignmentCreateNestedManyWithoutTopicInput
  }

  export type TopicUncheckedCreateInput = {
    id?: string
    name: string
    description?: string | null
    subtopic?: string | null
    isSystemTopic?: boolean
    isActive?: boolean
    embedding?: TopicCreateembeddingInput | number[]
    createdAt?: Date | string
    updatedAt?: Date | string
    topicAssignments?: TopicAssignmentUncheckedCreateNestedManyWithoutTopicInput
  }

  export type TopicUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: NullableStringFieldUpdateOperationsInput | string | null
    subtopic?: NullableStringFieldUpdateOperationsInput | string | null
    isSystemTopic?: BoolFieldUpdateOperationsInput | boolean
    isActive?: BoolFieldUpdateOperationsInput | boolean
    embedding?: TopicUpdateembeddingInput | number[]
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    topicAssignments?: TopicAssignmentUpdateManyWithoutTopicNestedInput
  }

  export type TopicUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: NullableStringFieldUpdateOperationsInput | string | null
    subtopic?: NullableStringFieldUpdateOperationsInput | string | null
    isSystemTopic?: BoolFieldUpdateOperationsInput | boolean
    isActive?: BoolFieldUpdateOperationsInput | boolean
    embedding?: TopicUpdateembeddingInput | number[]
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    topicAssignments?: TopicAssignmentUncheckedUpdateManyWithoutTopicNestedInput
  }

  export type TopicCreateManyInput = {
    id?: string
    name: string
    description?: string | null
    subtopic?: string | null
    isSystemTopic?: boolean
    isActive?: boolean
    embedding?: TopicCreateembeddingInput | number[]
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type TopicUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: NullableStringFieldUpdateOperationsInput | string | null
    subtopic?: NullableStringFieldUpdateOperationsInput | string | null
    isSystemTopic?: BoolFieldUpdateOperationsInput | boolean
    isActive?: BoolFieldUpdateOperationsInput | boolean
    embedding?: TopicUpdateembeddingInput | number[]
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: NullableStringFieldUpdateOperationsInput | string | null
    subtopic?: NullableStringFieldUpdateOperationsInput | string | null
    isSystemTopic?: BoolFieldUpdateOperationsInput | boolean
    isActive?: BoolFieldUpdateOperationsInput | boolean
    embedding?: TopicUpdateembeddingInput | number[]
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicAssignmentCreateInput = {
    id?: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    createdAt?: Date | string
    question: QuestionCreateNestedOneWithoutTopicAssignmentsInput
    topic: TopicCreateNestedOneWithoutTopicAssignmentsInput
    analysisRun: AnalysisRunCreateNestedOneWithoutTopicAssignmentsInput
  }

  export type TopicAssignmentUncheckedCreateInput = {
    id?: string
    questionId: string
    topicId: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type TopicAssignmentUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    question?: QuestionUpdateOneRequiredWithoutTopicAssignmentsNestedInput
    topic?: TopicUpdateOneRequiredWithoutTopicAssignmentsNestedInput
    analysisRun?: AnalysisRunUpdateOneRequiredWithoutTopicAssignmentsNestedInput
  }

  export type TopicAssignmentUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    topicId?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicAssignmentCreateManyInput = {
    id?: string
    questionId: string
    topicId: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type TopicAssignmentUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicAssignmentUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    topicId?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ClusterResultCreateInput = {
    id?: string
    clusterId: number
    clusterName?: string | null
    createdAt?: Date | string
    question: QuestionCreateNestedOneWithoutClusterResultsInput
    analysisRun: AnalysisRunCreateNestedOneWithoutClusterResultsInput
  }

  export type ClusterResultUncheckedCreateInput = {
    id?: string
    questionId: string
    clusterId: number
    clusterName?: string | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type ClusterResultUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    question?: QuestionUpdateOneRequiredWithoutClusterResultsNestedInput
    analysisRun?: AnalysisRunUpdateOneRequiredWithoutClusterResultsNestedInput
  }

  export type ClusterResultUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ClusterResultCreateManyInput = {
    id?: string
    questionId: string
    clusterId: number
    clusterName?: string | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type ClusterResultUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ClusterResultUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type AnalysisRunCreateInput = {
    id?: string
    status?: $Enums.AnalysisStatus
    mode: string
    sampleSize?: number | null
    similarityThreshold: number
    embeddingModel: string
    gptModel: string
    totalQuestions?: number | null
    matchedQuestions?: number | null
    newTopicsFound?: number | null
    processingTimeMs?: number | null
    errorMessage?: string | null
    startedAt?: Date | string
    completedAt?: Date | string | null
    createdBy?: string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    topicAssignments?: TopicAssignmentCreateNestedManyWithoutAnalysisRunInput
    clusterResults?: ClusterResultCreateNestedManyWithoutAnalysisRunInput
  }

  export type AnalysisRunUncheckedCreateInput = {
    id?: string
    status?: $Enums.AnalysisStatus
    mode: string
    sampleSize?: number | null
    similarityThreshold: number
    embeddingModel: string
    gptModel: string
    totalQuestions?: number | null
    matchedQuestions?: number | null
    newTopicsFound?: number | null
    processingTimeMs?: number | null
    errorMessage?: string | null
    startedAt?: Date | string
    completedAt?: Date | string | null
    createdBy?: string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    topicAssignments?: TopicAssignmentUncheckedCreateNestedManyWithoutAnalysisRunInput
    clusterResults?: ClusterResultUncheckedCreateNestedManyWithoutAnalysisRunInput
  }

  export type AnalysisRunUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    status?: EnumAnalysisStatusFieldUpdateOperationsInput | $Enums.AnalysisStatus
    mode?: StringFieldUpdateOperationsInput | string
    sampleSize?: NullableIntFieldUpdateOperationsInput | number | null
    similarityThreshold?: FloatFieldUpdateOperationsInput | number
    embeddingModel?: StringFieldUpdateOperationsInput | string
    gptModel?: StringFieldUpdateOperationsInput | string
    totalQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    matchedQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    newTopicsFound?: NullableIntFieldUpdateOperationsInput | number | null
    processingTimeMs?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
    startedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    completedAt?: NullableDateTimeFieldUpdateOperationsInput | Date | string | null
    createdBy?: NullableStringFieldUpdateOperationsInput | string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    topicAssignments?: TopicAssignmentUpdateManyWithoutAnalysisRunNestedInput
    clusterResults?: ClusterResultUpdateManyWithoutAnalysisRunNestedInput
  }

  export type AnalysisRunUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    status?: EnumAnalysisStatusFieldUpdateOperationsInput | $Enums.AnalysisStatus
    mode?: StringFieldUpdateOperationsInput | string
    sampleSize?: NullableIntFieldUpdateOperationsInput | number | null
    similarityThreshold?: FloatFieldUpdateOperationsInput | number
    embeddingModel?: StringFieldUpdateOperationsInput | string
    gptModel?: StringFieldUpdateOperationsInput | string
    totalQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    matchedQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    newTopicsFound?: NullableIntFieldUpdateOperationsInput | number | null
    processingTimeMs?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
    startedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    completedAt?: NullableDateTimeFieldUpdateOperationsInput | Date | string | null
    createdBy?: NullableStringFieldUpdateOperationsInput | string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    topicAssignments?: TopicAssignmentUncheckedUpdateManyWithoutAnalysisRunNestedInput
    clusterResults?: ClusterResultUncheckedUpdateManyWithoutAnalysisRunNestedInput
  }

  export type AnalysisRunCreateManyInput = {
    id?: string
    status?: $Enums.AnalysisStatus
    mode: string
    sampleSize?: number | null
    similarityThreshold: number
    embeddingModel: string
    gptModel: string
    totalQuestions?: number | null
    matchedQuestions?: number | null
    newTopicsFound?: number | null
    processingTimeMs?: number | null
    errorMessage?: string | null
    startedAt?: Date | string
    completedAt?: Date | string | null
    createdBy?: string | null
    config?: NullableJsonNullValueInput | InputJsonValue
  }

  export type AnalysisRunUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    status?: EnumAnalysisStatusFieldUpdateOperationsInput | $Enums.AnalysisStatus
    mode?: StringFieldUpdateOperationsInput | string
    sampleSize?: NullableIntFieldUpdateOperationsInput | number | null
    similarityThreshold?: FloatFieldUpdateOperationsInput | number
    embeddingModel?: StringFieldUpdateOperationsInput | string
    gptModel?: StringFieldUpdateOperationsInput | string
    totalQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    matchedQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    newTopicsFound?: NullableIntFieldUpdateOperationsInput | number | null
    processingTimeMs?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
    startedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    completedAt?: NullableDateTimeFieldUpdateOperationsInput | Date | string | null
    createdBy?: NullableStringFieldUpdateOperationsInput | string | null
    config?: NullableJsonNullValueInput | InputJsonValue
  }

  export type AnalysisRunUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    status?: EnumAnalysisStatusFieldUpdateOperationsInput | $Enums.AnalysisStatus
    mode?: StringFieldUpdateOperationsInput | string
    sampleSize?: NullableIntFieldUpdateOperationsInput | number | null
    similarityThreshold?: FloatFieldUpdateOperationsInput | number
    embeddingModel?: StringFieldUpdateOperationsInput | string
    gptModel?: StringFieldUpdateOperationsInput | string
    totalQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    matchedQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    newTopicsFound?: NullableIntFieldUpdateOperationsInput | number | null
    processingTimeMs?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
    startedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    completedAt?: NullableDateTimeFieldUpdateOperationsInput | Date | string | null
    createdBy?: NullableStringFieldUpdateOperationsInput | string | null
    config?: NullableJsonNullValueInput | InputJsonValue
  }

  export type SystemConfigCreateInput = {
    id?: string
    key: string
    value: string
    type?: $Enums.ConfigType
    updatedAt?: Date | string
    updatedBy?: string | null
  }

  export type SystemConfigUncheckedCreateInput = {
    id?: string
    key: string
    value: string
    type?: $Enums.ConfigType
    updatedAt?: Date | string
    updatedBy?: string | null
  }

  export type SystemConfigUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    key?: StringFieldUpdateOperationsInput | string
    value?: StringFieldUpdateOperationsInput | string
    type?: EnumConfigTypeFieldUpdateOperationsInput | $Enums.ConfigType
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedBy?: NullableStringFieldUpdateOperationsInput | string | null
  }

  export type SystemConfigUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    key?: StringFieldUpdateOperationsInput | string
    value?: StringFieldUpdateOperationsInput | string
    type?: EnumConfigTypeFieldUpdateOperationsInput | $Enums.ConfigType
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedBy?: NullableStringFieldUpdateOperationsInput | string | null
  }

  export type SystemConfigCreateManyInput = {
    id?: string
    key: string
    value: string
    type?: $Enums.ConfigType
    updatedAt?: Date | string
    updatedBy?: string | null
  }

  export type SystemConfigUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    key?: StringFieldUpdateOperationsInput | string
    value?: StringFieldUpdateOperationsInput | string
    type?: EnumConfigTypeFieldUpdateOperationsInput | $Enums.ConfigType
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedBy?: NullableStringFieldUpdateOperationsInput | string | null
  }

  export type SystemConfigUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    key?: StringFieldUpdateOperationsInput | string
    value?: StringFieldUpdateOperationsInput | string
    type?: EnumConfigTypeFieldUpdateOperationsInput | $Enums.ConfigType
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedBy?: NullableStringFieldUpdateOperationsInput | string | null
  }

  export type UserSessionCreateInput = {
    id?: string
    sessionId: string
    userId?: string | null
    isDevSession?: boolean
    ipAddress?: string | null
    userAgent?: string | null
    country?: string | null
    region?: string | null
    createdAt?: Date | string
    lastActivity?: Date | string
  }

  export type UserSessionUncheckedCreateInput = {
    id?: string
    sessionId: string
    userId?: string | null
    isDevSession?: boolean
    ipAddress?: string | null
    userAgent?: string | null
    country?: string | null
    region?: string | null
    createdAt?: Date | string
    lastActivity?: Date | string
  }

  export type UserSessionUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    sessionId?: StringFieldUpdateOperationsInput | string
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    isDevSession?: BoolFieldUpdateOperationsInput | boolean
    ipAddress?: NullableStringFieldUpdateOperationsInput | string | null
    userAgent?: NullableStringFieldUpdateOperationsInput | string | null
    country?: NullableStringFieldUpdateOperationsInput | string | null
    region?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    lastActivity?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type UserSessionUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    sessionId?: StringFieldUpdateOperationsInput | string
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    isDevSession?: BoolFieldUpdateOperationsInput | boolean
    ipAddress?: NullableStringFieldUpdateOperationsInput | string | null
    userAgent?: NullableStringFieldUpdateOperationsInput | string | null
    country?: NullableStringFieldUpdateOperationsInput | string | null
    region?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    lastActivity?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type UserSessionCreateManyInput = {
    id?: string
    sessionId: string
    userId?: string | null
    isDevSession?: boolean
    ipAddress?: string | null
    userAgent?: string | null
    country?: string | null
    region?: string | null
    createdAt?: Date | string
    lastActivity?: Date | string
  }

  export type UserSessionUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    sessionId?: StringFieldUpdateOperationsInput | string
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    isDevSession?: BoolFieldUpdateOperationsInput | boolean
    ipAddress?: NullableStringFieldUpdateOperationsInput | string | null
    userAgent?: NullableStringFieldUpdateOperationsInput | string | null
    country?: NullableStringFieldUpdateOperationsInput | string | null
    region?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    lastActivity?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type UserSessionUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    sessionId?: StringFieldUpdateOperationsInput | string
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    isDevSession?: BoolFieldUpdateOperationsInput | boolean
    ipAddress?: NullableStringFieldUpdateOperationsInput | string | null
    userAgent?: NullableStringFieldUpdateOperationsInput | string | null
    country?: NullableStringFieldUpdateOperationsInput | string | null
    region?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    lastActivity?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DataSyncCreateInput = {
    id?: string
    source: string
    sourceUrl?: string | null
    lastSyncAt: Date | string
    status?: $Enums.SyncStatus
    recordsProcessed?: number | null
    recordsAdded?: number | null
    recordsUpdated?: number | null
    errorMessage?: string | null
  }

  export type DataSyncUncheckedCreateInput = {
    id?: string
    source: string
    sourceUrl?: string | null
    lastSyncAt: Date | string
    status?: $Enums.SyncStatus
    recordsProcessed?: number | null
    recordsAdded?: number | null
    recordsUpdated?: number | null
    errorMessage?: string | null
  }

  export type DataSyncUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    source?: StringFieldUpdateOperationsInput | string
    sourceUrl?: NullableStringFieldUpdateOperationsInput | string | null
    lastSyncAt?: DateTimeFieldUpdateOperationsInput | Date | string
    status?: EnumSyncStatusFieldUpdateOperationsInput | $Enums.SyncStatus
    recordsProcessed?: NullableIntFieldUpdateOperationsInput | number | null
    recordsAdded?: NullableIntFieldUpdateOperationsInput | number | null
    recordsUpdated?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
  }

  export type DataSyncUncheckedUpdateInput = {
    id?: StringFieldUpdateOperationsInput | string
    source?: StringFieldUpdateOperationsInput | string
    sourceUrl?: NullableStringFieldUpdateOperationsInput | string | null
    lastSyncAt?: DateTimeFieldUpdateOperationsInput | Date | string
    status?: EnumSyncStatusFieldUpdateOperationsInput | $Enums.SyncStatus
    recordsProcessed?: NullableIntFieldUpdateOperationsInput | number | null
    recordsAdded?: NullableIntFieldUpdateOperationsInput | number | null
    recordsUpdated?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
  }

  export type DataSyncCreateManyInput = {
    id?: string
    source: string
    sourceUrl?: string | null
    lastSyncAt: Date | string
    status?: $Enums.SyncStatus
    recordsProcessed?: number | null
    recordsAdded?: number | null
    recordsUpdated?: number | null
    errorMessage?: string | null
  }

  export type DataSyncUpdateManyMutationInput = {
    id?: StringFieldUpdateOperationsInput | string
    source?: StringFieldUpdateOperationsInput | string
    sourceUrl?: NullableStringFieldUpdateOperationsInput | string | null
    lastSyncAt?: DateTimeFieldUpdateOperationsInput | Date | string
    status?: EnumSyncStatusFieldUpdateOperationsInput | $Enums.SyncStatus
    recordsProcessed?: NullableIntFieldUpdateOperationsInput | number | null
    recordsAdded?: NullableIntFieldUpdateOperationsInput | number | null
    recordsUpdated?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
  }

  export type DataSyncUncheckedUpdateManyInput = {
    id?: StringFieldUpdateOperationsInput | string
    source?: StringFieldUpdateOperationsInput | string
    sourceUrl?: NullableStringFieldUpdateOperationsInput | string | null
    lastSyncAt?: DateTimeFieldUpdateOperationsInput | Date | string
    status?: EnumSyncStatusFieldUpdateOperationsInput | $Enums.SyncStatus
    recordsProcessed?: NullableIntFieldUpdateOperationsInput | number | null
    recordsAdded?: NullableIntFieldUpdateOperationsInput | number | null
    recordsUpdated?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
  }

  export type StringFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringFilter<$PrismaModel> | string
  }

  export type StringNullableFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringNullableFilter<$PrismaModel> | string | null
  }

  export type FloatNullableListFilter<$PrismaModel = never> = {
    equals?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    has?: number | FloatFieldRefInput<$PrismaModel> | null
    hasEvery?: number[] | ListFloatFieldRefInput<$PrismaModel>
    hasSome?: number[] | ListFloatFieldRefInput<$PrismaModel>
    isEmpty?: boolean
  }

  export type DateTimeFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeFilter<$PrismaModel> | Date | string
  }

  export type TopicAssignmentListRelationFilter = {
    every?: TopicAssignmentWhereInput
    some?: TopicAssignmentWhereInput
    none?: TopicAssignmentWhereInput
  }

  export type ClusterResultListRelationFilter = {
    every?: ClusterResultWhereInput
    some?: ClusterResultWhereInput
    none?: ClusterResultWhereInput
  }

  export type SortOrderInput = {
    sort: SortOrder
    nulls?: NullsOrder
  }

  export type TopicAssignmentOrderByRelationAggregateInput = {
    _count?: SortOrder
  }

  export type ClusterResultOrderByRelationAggregateInput = {
    _count?: SortOrder
  }

  export type QuestionCountOrderByAggregateInput = {
    id?: SortOrder
    text?: SortOrder
    originalText?: SortOrder
    embedding?: SortOrder
    language?: SortOrder
    country?: SortOrder
    state?: SortOrder
    userId?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type QuestionAvgOrderByAggregateInput = {
    embedding?: SortOrder
  }

  export type QuestionMaxOrderByAggregateInput = {
    id?: SortOrder
    text?: SortOrder
    originalText?: SortOrder
    language?: SortOrder
    country?: SortOrder
    state?: SortOrder
    userId?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type QuestionMinOrderByAggregateInput = {
    id?: SortOrder
    text?: SortOrder
    originalText?: SortOrder
    language?: SortOrder
    country?: SortOrder
    state?: SortOrder
    userId?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type QuestionSumOrderByAggregateInput = {
    embedding?: SortOrder
  }

  export type StringWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringWithAggregatesFilter<$PrismaModel> | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedStringFilter<$PrismaModel>
    _max?: NestedStringFilter<$PrismaModel>
  }

  export type StringNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringNullableWithAggregatesFilter<$PrismaModel> | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedStringNullableFilter<$PrismaModel>
    _max?: NestedStringNullableFilter<$PrismaModel>
  }

  export type DateTimeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeWithAggregatesFilter<$PrismaModel> | Date | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedDateTimeFilter<$PrismaModel>
    _max?: NestedDateTimeFilter<$PrismaModel>
  }

  export type BoolFilter<$PrismaModel = never> = {
    equals?: boolean | BooleanFieldRefInput<$PrismaModel>
    not?: NestedBoolFilter<$PrismaModel> | boolean
  }

  export type TopicCountOrderByAggregateInput = {
    id?: SortOrder
    name?: SortOrder
    description?: SortOrder
    subtopic?: SortOrder
    isSystemTopic?: SortOrder
    isActive?: SortOrder
    embedding?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type TopicAvgOrderByAggregateInput = {
    embedding?: SortOrder
  }

  export type TopicMaxOrderByAggregateInput = {
    id?: SortOrder
    name?: SortOrder
    description?: SortOrder
    subtopic?: SortOrder
    isSystemTopic?: SortOrder
    isActive?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type TopicMinOrderByAggregateInput = {
    id?: SortOrder
    name?: SortOrder
    description?: SortOrder
    subtopic?: SortOrder
    isSystemTopic?: SortOrder
    isActive?: SortOrder
    createdAt?: SortOrder
    updatedAt?: SortOrder
  }

  export type TopicSumOrderByAggregateInput = {
    embedding?: SortOrder
  }

  export type BoolWithAggregatesFilter<$PrismaModel = never> = {
    equals?: boolean | BooleanFieldRefInput<$PrismaModel>
    not?: NestedBoolWithAggregatesFilter<$PrismaModel> | boolean
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedBoolFilter<$PrismaModel>
    _max?: NestedBoolFilter<$PrismaModel>
  }

  export type FloatNullableFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel> | null
    in?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatNullableFilter<$PrismaModel> | number | null
  }

  export type EnumAssignmentTypeFilter<$PrismaModel = never> = {
    equals?: $Enums.AssignmentType | EnumAssignmentTypeFieldRefInput<$PrismaModel>
    in?: $Enums.AssignmentType[] | ListEnumAssignmentTypeFieldRefInput<$PrismaModel>
    notIn?: $Enums.AssignmentType[] | ListEnumAssignmentTypeFieldRefInput<$PrismaModel>
    not?: NestedEnumAssignmentTypeFilter<$PrismaModel> | $Enums.AssignmentType
  }

  export type QuestionScalarRelationFilter = {
    is?: QuestionWhereInput
    isNot?: QuestionWhereInput
  }

  export type TopicScalarRelationFilter = {
    is?: TopicWhereInput
    isNot?: TopicWhereInput
  }

  export type AnalysisRunScalarRelationFilter = {
    is?: AnalysisRunWhereInput
    isNot?: AnalysisRunWhereInput
  }

  export type TopicAssignmentQuestionIdTopicIdAnalysisRunIdCompoundUniqueInput = {
    questionId: string
    topicId: string
    analysisRunId: string
  }

  export type TopicAssignmentCountOrderByAggregateInput = {
    id?: SortOrder
    questionId?: SortOrder
    topicId?: SortOrder
    similarityScore?: SortOrder
    assignmentType?: SortOrder
    confidence?: SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
  }

  export type TopicAssignmentAvgOrderByAggregateInput = {
    similarityScore?: SortOrder
    confidence?: SortOrder
  }

  export type TopicAssignmentMaxOrderByAggregateInput = {
    id?: SortOrder
    questionId?: SortOrder
    topicId?: SortOrder
    similarityScore?: SortOrder
    assignmentType?: SortOrder
    confidence?: SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
  }

  export type TopicAssignmentMinOrderByAggregateInput = {
    id?: SortOrder
    questionId?: SortOrder
    topicId?: SortOrder
    similarityScore?: SortOrder
    assignmentType?: SortOrder
    confidence?: SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
  }

  export type TopicAssignmentSumOrderByAggregateInput = {
    similarityScore?: SortOrder
    confidence?: SortOrder
  }

  export type FloatNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel> | null
    in?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatNullableWithAggregatesFilter<$PrismaModel> | number | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _avg?: NestedFloatNullableFilter<$PrismaModel>
    _sum?: NestedFloatNullableFilter<$PrismaModel>
    _min?: NestedFloatNullableFilter<$PrismaModel>
    _max?: NestedFloatNullableFilter<$PrismaModel>
  }

  export type EnumAssignmentTypeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.AssignmentType | EnumAssignmentTypeFieldRefInput<$PrismaModel>
    in?: $Enums.AssignmentType[] | ListEnumAssignmentTypeFieldRefInput<$PrismaModel>
    notIn?: $Enums.AssignmentType[] | ListEnumAssignmentTypeFieldRefInput<$PrismaModel>
    not?: NestedEnumAssignmentTypeWithAggregatesFilter<$PrismaModel> | $Enums.AssignmentType
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumAssignmentTypeFilter<$PrismaModel>
    _max?: NestedEnumAssignmentTypeFilter<$PrismaModel>
  }

  export type IntFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntFilter<$PrismaModel> | number
  }

  export type ClusterResultQuestionIdAnalysisRunIdCompoundUniqueInput = {
    questionId: string
    analysisRunId: string
  }

  export type ClusterResultCountOrderByAggregateInput = {
    id?: SortOrder
    questionId?: SortOrder
    clusterId?: SortOrder
    clusterName?: SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
  }

  export type ClusterResultAvgOrderByAggregateInput = {
    clusterId?: SortOrder
  }

  export type ClusterResultMaxOrderByAggregateInput = {
    id?: SortOrder
    questionId?: SortOrder
    clusterId?: SortOrder
    clusterName?: SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
  }

  export type ClusterResultMinOrderByAggregateInput = {
    id?: SortOrder
    questionId?: SortOrder
    clusterId?: SortOrder
    clusterName?: SortOrder
    analysisRunId?: SortOrder
    createdAt?: SortOrder
  }

  export type ClusterResultSumOrderByAggregateInput = {
    clusterId?: SortOrder
  }

  export type IntWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedIntFilter<$PrismaModel>
    _min?: NestedIntFilter<$PrismaModel>
    _max?: NestedIntFilter<$PrismaModel>
  }

  export type EnumAnalysisStatusFilter<$PrismaModel = never> = {
    equals?: $Enums.AnalysisStatus | EnumAnalysisStatusFieldRefInput<$PrismaModel>
    in?: $Enums.AnalysisStatus[] | ListEnumAnalysisStatusFieldRefInput<$PrismaModel>
    notIn?: $Enums.AnalysisStatus[] | ListEnumAnalysisStatusFieldRefInput<$PrismaModel>
    not?: NestedEnumAnalysisStatusFilter<$PrismaModel> | $Enums.AnalysisStatus
  }

  export type IntNullableFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableFilter<$PrismaModel> | number | null
  }

  export type FloatFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatFilter<$PrismaModel> | number
  }

  export type DateTimeNullableFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel> | null
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel> | null
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel> | null
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeNullableFilter<$PrismaModel> | Date | string | null
  }
  export type JsonNullableFilter<$PrismaModel = never> =
    | PatchUndefined<
        Either<Required<JsonNullableFilterBase<$PrismaModel>>, Exclude<keyof Required<JsonNullableFilterBase<$PrismaModel>>, 'path'>>,
        Required<JsonNullableFilterBase<$PrismaModel>>
      >
    | OptionalFlat<Omit<Required<JsonNullableFilterBase<$PrismaModel>>, 'path'>>

  export type JsonNullableFilterBase<$PrismaModel = never> = {
    equals?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
    path?: string[]
    mode?: QueryMode | EnumQueryModeFieldRefInput<$PrismaModel>
    string_contains?: string | StringFieldRefInput<$PrismaModel>
    string_starts_with?: string | StringFieldRefInput<$PrismaModel>
    string_ends_with?: string | StringFieldRefInput<$PrismaModel>
    array_starts_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_ends_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_contains?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    lt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    lte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    not?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
  }

  export type AnalysisRunCountOrderByAggregateInput = {
    id?: SortOrder
    status?: SortOrder
    mode?: SortOrder
    sampleSize?: SortOrder
    similarityThreshold?: SortOrder
    embeddingModel?: SortOrder
    gptModel?: SortOrder
    totalQuestions?: SortOrder
    matchedQuestions?: SortOrder
    newTopicsFound?: SortOrder
    processingTimeMs?: SortOrder
    errorMessage?: SortOrder
    startedAt?: SortOrder
    completedAt?: SortOrder
    createdBy?: SortOrder
    config?: SortOrder
  }

  export type AnalysisRunAvgOrderByAggregateInput = {
    sampleSize?: SortOrder
    similarityThreshold?: SortOrder
    totalQuestions?: SortOrder
    matchedQuestions?: SortOrder
    newTopicsFound?: SortOrder
    processingTimeMs?: SortOrder
  }

  export type AnalysisRunMaxOrderByAggregateInput = {
    id?: SortOrder
    status?: SortOrder
    mode?: SortOrder
    sampleSize?: SortOrder
    similarityThreshold?: SortOrder
    embeddingModel?: SortOrder
    gptModel?: SortOrder
    totalQuestions?: SortOrder
    matchedQuestions?: SortOrder
    newTopicsFound?: SortOrder
    processingTimeMs?: SortOrder
    errorMessage?: SortOrder
    startedAt?: SortOrder
    completedAt?: SortOrder
    createdBy?: SortOrder
  }

  export type AnalysisRunMinOrderByAggregateInput = {
    id?: SortOrder
    status?: SortOrder
    mode?: SortOrder
    sampleSize?: SortOrder
    similarityThreshold?: SortOrder
    embeddingModel?: SortOrder
    gptModel?: SortOrder
    totalQuestions?: SortOrder
    matchedQuestions?: SortOrder
    newTopicsFound?: SortOrder
    processingTimeMs?: SortOrder
    errorMessage?: SortOrder
    startedAt?: SortOrder
    completedAt?: SortOrder
    createdBy?: SortOrder
  }

  export type AnalysisRunSumOrderByAggregateInput = {
    sampleSize?: SortOrder
    similarityThreshold?: SortOrder
    totalQuestions?: SortOrder
    matchedQuestions?: SortOrder
    newTopicsFound?: SortOrder
    processingTimeMs?: SortOrder
  }

  export type EnumAnalysisStatusWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.AnalysisStatus | EnumAnalysisStatusFieldRefInput<$PrismaModel>
    in?: $Enums.AnalysisStatus[] | ListEnumAnalysisStatusFieldRefInput<$PrismaModel>
    notIn?: $Enums.AnalysisStatus[] | ListEnumAnalysisStatusFieldRefInput<$PrismaModel>
    not?: NestedEnumAnalysisStatusWithAggregatesFilter<$PrismaModel> | $Enums.AnalysisStatus
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumAnalysisStatusFilter<$PrismaModel>
    _max?: NestedEnumAnalysisStatusFilter<$PrismaModel>
  }

  export type IntNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableWithAggregatesFilter<$PrismaModel> | number | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _avg?: NestedFloatNullableFilter<$PrismaModel>
    _sum?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedIntNullableFilter<$PrismaModel>
    _max?: NestedIntNullableFilter<$PrismaModel>
  }

  export type FloatWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedFloatFilter<$PrismaModel>
    _min?: NestedFloatFilter<$PrismaModel>
    _max?: NestedFloatFilter<$PrismaModel>
  }

  export type DateTimeNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel> | null
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel> | null
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel> | null
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeNullableWithAggregatesFilter<$PrismaModel> | Date | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedDateTimeNullableFilter<$PrismaModel>
    _max?: NestedDateTimeNullableFilter<$PrismaModel>
  }
  export type JsonNullableWithAggregatesFilter<$PrismaModel = never> =
    | PatchUndefined<
        Either<Required<JsonNullableWithAggregatesFilterBase<$PrismaModel>>, Exclude<keyof Required<JsonNullableWithAggregatesFilterBase<$PrismaModel>>, 'path'>>,
        Required<JsonNullableWithAggregatesFilterBase<$PrismaModel>>
      >
    | OptionalFlat<Omit<Required<JsonNullableWithAggregatesFilterBase<$PrismaModel>>, 'path'>>

  export type JsonNullableWithAggregatesFilterBase<$PrismaModel = never> = {
    equals?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
    path?: string[]
    mode?: QueryMode | EnumQueryModeFieldRefInput<$PrismaModel>
    string_contains?: string | StringFieldRefInput<$PrismaModel>
    string_starts_with?: string | StringFieldRefInput<$PrismaModel>
    string_ends_with?: string | StringFieldRefInput<$PrismaModel>
    array_starts_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_ends_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_contains?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    lt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    lte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    not?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedJsonNullableFilter<$PrismaModel>
    _max?: NestedJsonNullableFilter<$PrismaModel>
  }

  export type EnumConfigTypeFilter<$PrismaModel = never> = {
    equals?: $Enums.ConfigType | EnumConfigTypeFieldRefInput<$PrismaModel>
    in?: $Enums.ConfigType[] | ListEnumConfigTypeFieldRefInput<$PrismaModel>
    notIn?: $Enums.ConfigType[] | ListEnumConfigTypeFieldRefInput<$PrismaModel>
    not?: NestedEnumConfigTypeFilter<$PrismaModel> | $Enums.ConfigType
  }

  export type SystemConfigCountOrderByAggregateInput = {
    id?: SortOrder
    key?: SortOrder
    value?: SortOrder
    type?: SortOrder
    updatedAt?: SortOrder
    updatedBy?: SortOrder
  }

  export type SystemConfigMaxOrderByAggregateInput = {
    id?: SortOrder
    key?: SortOrder
    value?: SortOrder
    type?: SortOrder
    updatedAt?: SortOrder
    updatedBy?: SortOrder
  }

  export type SystemConfigMinOrderByAggregateInput = {
    id?: SortOrder
    key?: SortOrder
    value?: SortOrder
    type?: SortOrder
    updatedAt?: SortOrder
    updatedBy?: SortOrder
  }

  export type EnumConfigTypeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.ConfigType | EnumConfigTypeFieldRefInput<$PrismaModel>
    in?: $Enums.ConfigType[] | ListEnumConfigTypeFieldRefInput<$PrismaModel>
    notIn?: $Enums.ConfigType[] | ListEnumConfigTypeFieldRefInput<$PrismaModel>
    not?: NestedEnumConfigTypeWithAggregatesFilter<$PrismaModel> | $Enums.ConfigType
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumConfigTypeFilter<$PrismaModel>
    _max?: NestedEnumConfigTypeFilter<$PrismaModel>
  }

  export type UserSessionCountOrderByAggregateInput = {
    id?: SortOrder
    sessionId?: SortOrder
    userId?: SortOrder
    isDevSession?: SortOrder
    ipAddress?: SortOrder
    userAgent?: SortOrder
    country?: SortOrder
    region?: SortOrder
    createdAt?: SortOrder
    lastActivity?: SortOrder
  }

  export type UserSessionMaxOrderByAggregateInput = {
    id?: SortOrder
    sessionId?: SortOrder
    userId?: SortOrder
    isDevSession?: SortOrder
    ipAddress?: SortOrder
    userAgent?: SortOrder
    country?: SortOrder
    region?: SortOrder
    createdAt?: SortOrder
    lastActivity?: SortOrder
  }

  export type UserSessionMinOrderByAggregateInput = {
    id?: SortOrder
    sessionId?: SortOrder
    userId?: SortOrder
    isDevSession?: SortOrder
    ipAddress?: SortOrder
    userAgent?: SortOrder
    country?: SortOrder
    region?: SortOrder
    createdAt?: SortOrder
    lastActivity?: SortOrder
  }

  export type EnumSyncStatusFilter<$PrismaModel = never> = {
    equals?: $Enums.SyncStatus | EnumSyncStatusFieldRefInput<$PrismaModel>
    in?: $Enums.SyncStatus[] | ListEnumSyncStatusFieldRefInput<$PrismaModel>
    notIn?: $Enums.SyncStatus[] | ListEnumSyncStatusFieldRefInput<$PrismaModel>
    not?: NestedEnumSyncStatusFilter<$PrismaModel> | $Enums.SyncStatus
  }

  export type DataSyncCountOrderByAggregateInput = {
    id?: SortOrder
    source?: SortOrder
    sourceUrl?: SortOrder
    lastSyncAt?: SortOrder
    status?: SortOrder
    recordsProcessed?: SortOrder
    recordsAdded?: SortOrder
    recordsUpdated?: SortOrder
    errorMessage?: SortOrder
  }

  export type DataSyncAvgOrderByAggregateInput = {
    recordsProcessed?: SortOrder
    recordsAdded?: SortOrder
    recordsUpdated?: SortOrder
  }

  export type DataSyncMaxOrderByAggregateInput = {
    id?: SortOrder
    source?: SortOrder
    sourceUrl?: SortOrder
    lastSyncAt?: SortOrder
    status?: SortOrder
    recordsProcessed?: SortOrder
    recordsAdded?: SortOrder
    recordsUpdated?: SortOrder
    errorMessage?: SortOrder
  }

  export type DataSyncMinOrderByAggregateInput = {
    id?: SortOrder
    source?: SortOrder
    sourceUrl?: SortOrder
    lastSyncAt?: SortOrder
    status?: SortOrder
    recordsProcessed?: SortOrder
    recordsAdded?: SortOrder
    recordsUpdated?: SortOrder
    errorMessage?: SortOrder
  }

  export type DataSyncSumOrderByAggregateInput = {
    recordsProcessed?: SortOrder
    recordsAdded?: SortOrder
    recordsUpdated?: SortOrder
  }

  export type EnumSyncStatusWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.SyncStatus | EnumSyncStatusFieldRefInput<$PrismaModel>
    in?: $Enums.SyncStatus[] | ListEnumSyncStatusFieldRefInput<$PrismaModel>
    notIn?: $Enums.SyncStatus[] | ListEnumSyncStatusFieldRefInput<$PrismaModel>
    not?: NestedEnumSyncStatusWithAggregatesFilter<$PrismaModel> | $Enums.SyncStatus
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumSyncStatusFilter<$PrismaModel>
    _max?: NestedEnumSyncStatusFilter<$PrismaModel>
  }

  export type QuestionCreateembeddingInput = {
    set: number[]
  }

  export type TopicAssignmentCreateNestedManyWithoutQuestionInput = {
    create?: XOR<TopicAssignmentCreateWithoutQuestionInput, TopicAssignmentUncheckedCreateWithoutQuestionInput> | TopicAssignmentCreateWithoutQuestionInput[] | TopicAssignmentUncheckedCreateWithoutQuestionInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutQuestionInput | TopicAssignmentCreateOrConnectWithoutQuestionInput[]
    createMany?: TopicAssignmentCreateManyQuestionInputEnvelope
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
  }

  export type ClusterResultCreateNestedManyWithoutQuestionInput = {
    create?: XOR<ClusterResultCreateWithoutQuestionInput, ClusterResultUncheckedCreateWithoutQuestionInput> | ClusterResultCreateWithoutQuestionInput[] | ClusterResultUncheckedCreateWithoutQuestionInput[]
    connectOrCreate?: ClusterResultCreateOrConnectWithoutQuestionInput | ClusterResultCreateOrConnectWithoutQuestionInput[]
    createMany?: ClusterResultCreateManyQuestionInputEnvelope
    connect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
  }

  export type TopicAssignmentUncheckedCreateNestedManyWithoutQuestionInput = {
    create?: XOR<TopicAssignmentCreateWithoutQuestionInput, TopicAssignmentUncheckedCreateWithoutQuestionInput> | TopicAssignmentCreateWithoutQuestionInput[] | TopicAssignmentUncheckedCreateWithoutQuestionInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutQuestionInput | TopicAssignmentCreateOrConnectWithoutQuestionInput[]
    createMany?: TopicAssignmentCreateManyQuestionInputEnvelope
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
  }

  export type ClusterResultUncheckedCreateNestedManyWithoutQuestionInput = {
    create?: XOR<ClusterResultCreateWithoutQuestionInput, ClusterResultUncheckedCreateWithoutQuestionInput> | ClusterResultCreateWithoutQuestionInput[] | ClusterResultUncheckedCreateWithoutQuestionInput[]
    connectOrCreate?: ClusterResultCreateOrConnectWithoutQuestionInput | ClusterResultCreateOrConnectWithoutQuestionInput[]
    createMany?: ClusterResultCreateManyQuestionInputEnvelope
    connect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
  }

  export type StringFieldUpdateOperationsInput = {
    set?: string
  }

  export type NullableStringFieldUpdateOperationsInput = {
    set?: string | null
  }

  export type QuestionUpdateembeddingInput = {
    set?: number[]
    push?: number | number[]
  }

  export type DateTimeFieldUpdateOperationsInput = {
    set?: Date | string
  }

  export type TopicAssignmentUpdateManyWithoutQuestionNestedInput = {
    create?: XOR<TopicAssignmentCreateWithoutQuestionInput, TopicAssignmentUncheckedCreateWithoutQuestionInput> | TopicAssignmentCreateWithoutQuestionInput[] | TopicAssignmentUncheckedCreateWithoutQuestionInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutQuestionInput | TopicAssignmentCreateOrConnectWithoutQuestionInput[]
    upsert?: TopicAssignmentUpsertWithWhereUniqueWithoutQuestionInput | TopicAssignmentUpsertWithWhereUniqueWithoutQuestionInput[]
    createMany?: TopicAssignmentCreateManyQuestionInputEnvelope
    set?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    disconnect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    delete?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    update?: TopicAssignmentUpdateWithWhereUniqueWithoutQuestionInput | TopicAssignmentUpdateWithWhereUniqueWithoutQuestionInput[]
    updateMany?: TopicAssignmentUpdateManyWithWhereWithoutQuestionInput | TopicAssignmentUpdateManyWithWhereWithoutQuestionInput[]
    deleteMany?: TopicAssignmentScalarWhereInput | TopicAssignmentScalarWhereInput[]
  }

  export type ClusterResultUpdateManyWithoutQuestionNestedInput = {
    create?: XOR<ClusterResultCreateWithoutQuestionInput, ClusterResultUncheckedCreateWithoutQuestionInput> | ClusterResultCreateWithoutQuestionInput[] | ClusterResultUncheckedCreateWithoutQuestionInput[]
    connectOrCreate?: ClusterResultCreateOrConnectWithoutQuestionInput | ClusterResultCreateOrConnectWithoutQuestionInput[]
    upsert?: ClusterResultUpsertWithWhereUniqueWithoutQuestionInput | ClusterResultUpsertWithWhereUniqueWithoutQuestionInput[]
    createMany?: ClusterResultCreateManyQuestionInputEnvelope
    set?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    disconnect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    delete?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    connect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    update?: ClusterResultUpdateWithWhereUniqueWithoutQuestionInput | ClusterResultUpdateWithWhereUniqueWithoutQuestionInput[]
    updateMany?: ClusterResultUpdateManyWithWhereWithoutQuestionInput | ClusterResultUpdateManyWithWhereWithoutQuestionInput[]
    deleteMany?: ClusterResultScalarWhereInput | ClusterResultScalarWhereInput[]
  }

  export type TopicAssignmentUncheckedUpdateManyWithoutQuestionNestedInput = {
    create?: XOR<TopicAssignmentCreateWithoutQuestionInput, TopicAssignmentUncheckedCreateWithoutQuestionInput> | TopicAssignmentCreateWithoutQuestionInput[] | TopicAssignmentUncheckedCreateWithoutQuestionInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutQuestionInput | TopicAssignmentCreateOrConnectWithoutQuestionInput[]
    upsert?: TopicAssignmentUpsertWithWhereUniqueWithoutQuestionInput | TopicAssignmentUpsertWithWhereUniqueWithoutQuestionInput[]
    createMany?: TopicAssignmentCreateManyQuestionInputEnvelope
    set?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    disconnect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    delete?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    update?: TopicAssignmentUpdateWithWhereUniqueWithoutQuestionInput | TopicAssignmentUpdateWithWhereUniqueWithoutQuestionInput[]
    updateMany?: TopicAssignmentUpdateManyWithWhereWithoutQuestionInput | TopicAssignmentUpdateManyWithWhereWithoutQuestionInput[]
    deleteMany?: TopicAssignmentScalarWhereInput | TopicAssignmentScalarWhereInput[]
  }

  export type ClusterResultUncheckedUpdateManyWithoutQuestionNestedInput = {
    create?: XOR<ClusterResultCreateWithoutQuestionInput, ClusterResultUncheckedCreateWithoutQuestionInput> | ClusterResultCreateWithoutQuestionInput[] | ClusterResultUncheckedCreateWithoutQuestionInput[]
    connectOrCreate?: ClusterResultCreateOrConnectWithoutQuestionInput | ClusterResultCreateOrConnectWithoutQuestionInput[]
    upsert?: ClusterResultUpsertWithWhereUniqueWithoutQuestionInput | ClusterResultUpsertWithWhereUniqueWithoutQuestionInput[]
    createMany?: ClusterResultCreateManyQuestionInputEnvelope
    set?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    disconnect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    delete?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    connect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    update?: ClusterResultUpdateWithWhereUniqueWithoutQuestionInput | ClusterResultUpdateWithWhereUniqueWithoutQuestionInput[]
    updateMany?: ClusterResultUpdateManyWithWhereWithoutQuestionInput | ClusterResultUpdateManyWithWhereWithoutQuestionInput[]
    deleteMany?: ClusterResultScalarWhereInput | ClusterResultScalarWhereInput[]
  }

  export type TopicCreateembeddingInput = {
    set: number[]
  }

  export type TopicAssignmentCreateNestedManyWithoutTopicInput = {
    create?: XOR<TopicAssignmentCreateWithoutTopicInput, TopicAssignmentUncheckedCreateWithoutTopicInput> | TopicAssignmentCreateWithoutTopicInput[] | TopicAssignmentUncheckedCreateWithoutTopicInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutTopicInput | TopicAssignmentCreateOrConnectWithoutTopicInput[]
    createMany?: TopicAssignmentCreateManyTopicInputEnvelope
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
  }

  export type TopicAssignmentUncheckedCreateNestedManyWithoutTopicInput = {
    create?: XOR<TopicAssignmentCreateWithoutTopicInput, TopicAssignmentUncheckedCreateWithoutTopicInput> | TopicAssignmentCreateWithoutTopicInput[] | TopicAssignmentUncheckedCreateWithoutTopicInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutTopicInput | TopicAssignmentCreateOrConnectWithoutTopicInput[]
    createMany?: TopicAssignmentCreateManyTopicInputEnvelope
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
  }

  export type BoolFieldUpdateOperationsInput = {
    set?: boolean
  }

  export type TopicUpdateembeddingInput = {
    set?: number[]
    push?: number | number[]
  }

  export type TopicAssignmentUpdateManyWithoutTopicNestedInput = {
    create?: XOR<TopicAssignmentCreateWithoutTopicInput, TopicAssignmentUncheckedCreateWithoutTopicInput> | TopicAssignmentCreateWithoutTopicInput[] | TopicAssignmentUncheckedCreateWithoutTopicInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutTopicInput | TopicAssignmentCreateOrConnectWithoutTopicInput[]
    upsert?: TopicAssignmentUpsertWithWhereUniqueWithoutTopicInput | TopicAssignmentUpsertWithWhereUniqueWithoutTopicInput[]
    createMany?: TopicAssignmentCreateManyTopicInputEnvelope
    set?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    disconnect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    delete?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    update?: TopicAssignmentUpdateWithWhereUniqueWithoutTopicInput | TopicAssignmentUpdateWithWhereUniqueWithoutTopicInput[]
    updateMany?: TopicAssignmentUpdateManyWithWhereWithoutTopicInput | TopicAssignmentUpdateManyWithWhereWithoutTopicInput[]
    deleteMany?: TopicAssignmentScalarWhereInput | TopicAssignmentScalarWhereInput[]
  }

  export type TopicAssignmentUncheckedUpdateManyWithoutTopicNestedInput = {
    create?: XOR<TopicAssignmentCreateWithoutTopicInput, TopicAssignmentUncheckedCreateWithoutTopicInput> | TopicAssignmentCreateWithoutTopicInput[] | TopicAssignmentUncheckedCreateWithoutTopicInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutTopicInput | TopicAssignmentCreateOrConnectWithoutTopicInput[]
    upsert?: TopicAssignmentUpsertWithWhereUniqueWithoutTopicInput | TopicAssignmentUpsertWithWhereUniqueWithoutTopicInput[]
    createMany?: TopicAssignmentCreateManyTopicInputEnvelope
    set?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    disconnect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    delete?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    update?: TopicAssignmentUpdateWithWhereUniqueWithoutTopicInput | TopicAssignmentUpdateWithWhereUniqueWithoutTopicInput[]
    updateMany?: TopicAssignmentUpdateManyWithWhereWithoutTopicInput | TopicAssignmentUpdateManyWithWhereWithoutTopicInput[]
    deleteMany?: TopicAssignmentScalarWhereInput | TopicAssignmentScalarWhereInput[]
  }

  export type QuestionCreateNestedOneWithoutTopicAssignmentsInput = {
    create?: XOR<QuestionCreateWithoutTopicAssignmentsInput, QuestionUncheckedCreateWithoutTopicAssignmentsInput>
    connectOrCreate?: QuestionCreateOrConnectWithoutTopicAssignmentsInput
    connect?: QuestionWhereUniqueInput
  }

  export type TopicCreateNestedOneWithoutTopicAssignmentsInput = {
    create?: XOR<TopicCreateWithoutTopicAssignmentsInput, TopicUncheckedCreateWithoutTopicAssignmentsInput>
    connectOrCreate?: TopicCreateOrConnectWithoutTopicAssignmentsInput
    connect?: TopicWhereUniqueInput
  }

  export type AnalysisRunCreateNestedOneWithoutTopicAssignmentsInput = {
    create?: XOR<AnalysisRunCreateWithoutTopicAssignmentsInput, AnalysisRunUncheckedCreateWithoutTopicAssignmentsInput>
    connectOrCreate?: AnalysisRunCreateOrConnectWithoutTopicAssignmentsInput
    connect?: AnalysisRunWhereUniqueInput
  }

  export type NullableFloatFieldUpdateOperationsInput = {
    set?: number | null
    increment?: number
    decrement?: number
    multiply?: number
    divide?: number
  }

  export type EnumAssignmentTypeFieldUpdateOperationsInput = {
    set?: $Enums.AssignmentType
  }

  export type QuestionUpdateOneRequiredWithoutTopicAssignmentsNestedInput = {
    create?: XOR<QuestionCreateWithoutTopicAssignmentsInput, QuestionUncheckedCreateWithoutTopicAssignmentsInput>
    connectOrCreate?: QuestionCreateOrConnectWithoutTopicAssignmentsInput
    upsert?: QuestionUpsertWithoutTopicAssignmentsInput
    connect?: QuestionWhereUniqueInput
    update?: XOR<XOR<QuestionUpdateToOneWithWhereWithoutTopicAssignmentsInput, QuestionUpdateWithoutTopicAssignmentsInput>, QuestionUncheckedUpdateWithoutTopicAssignmentsInput>
  }

  export type TopicUpdateOneRequiredWithoutTopicAssignmentsNestedInput = {
    create?: XOR<TopicCreateWithoutTopicAssignmentsInput, TopicUncheckedCreateWithoutTopicAssignmentsInput>
    connectOrCreate?: TopicCreateOrConnectWithoutTopicAssignmentsInput
    upsert?: TopicUpsertWithoutTopicAssignmentsInput
    connect?: TopicWhereUniqueInput
    update?: XOR<XOR<TopicUpdateToOneWithWhereWithoutTopicAssignmentsInput, TopicUpdateWithoutTopicAssignmentsInput>, TopicUncheckedUpdateWithoutTopicAssignmentsInput>
  }

  export type AnalysisRunUpdateOneRequiredWithoutTopicAssignmentsNestedInput = {
    create?: XOR<AnalysisRunCreateWithoutTopicAssignmentsInput, AnalysisRunUncheckedCreateWithoutTopicAssignmentsInput>
    connectOrCreate?: AnalysisRunCreateOrConnectWithoutTopicAssignmentsInput
    upsert?: AnalysisRunUpsertWithoutTopicAssignmentsInput
    connect?: AnalysisRunWhereUniqueInput
    update?: XOR<XOR<AnalysisRunUpdateToOneWithWhereWithoutTopicAssignmentsInput, AnalysisRunUpdateWithoutTopicAssignmentsInput>, AnalysisRunUncheckedUpdateWithoutTopicAssignmentsInput>
  }

  export type QuestionCreateNestedOneWithoutClusterResultsInput = {
    create?: XOR<QuestionCreateWithoutClusterResultsInput, QuestionUncheckedCreateWithoutClusterResultsInput>
    connectOrCreate?: QuestionCreateOrConnectWithoutClusterResultsInput
    connect?: QuestionWhereUniqueInput
  }

  export type AnalysisRunCreateNestedOneWithoutClusterResultsInput = {
    create?: XOR<AnalysisRunCreateWithoutClusterResultsInput, AnalysisRunUncheckedCreateWithoutClusterResultsInput>
    connectOrCreate?: AnalysisRunCreateOrConnectWithoutClusterResultsInput
    connect?: AnalysisRunWhereUniqueInput
  }

  export type IntFieldUpdateOperationsInput = {
    set?: number
    increment?: number
    decrement?: number
    multiply?: number
    divide?: number
  }

  export type QuestionUpdateOneRequiredWithoutClusterResultsNestedInput = {
    create?: XOR<QuestionCreateWithoutClusterResultsInput, QuestionUncheckedCreateWithoutClusterResultsInput>
    connectOrCreate?: QuestionCreateOrConnectWithoutClusterResultsInput
    upsert?: QuestionUpsertWithoutClusterResultsInput
    connect?: QuestionWhereUniqueInput
    update?: XOR<XOR<QuestionUpdateToOneWithWhereWithoutClusterResultsInput, QuestionUpdateWithoutClusterResultsInput>, QuestionUncheckedUpdateWithoutClusterResultsInput>
  }

  export type AnalysisRunUpdateOneRequiredWithoutClusterResultsNestedInput = {
    create?: XOR<AnalysisRunCreateWithoutClusterResultsInput, AnalysisRunUncheckedCreateWithoutClusterResultsInput>
    connectOrCreate?: AnalysisRunCreateOrConnectWithoutClusterResultsInput
    upsert?: AnalysisRunUpsertWithoutClusterResultsInput
    connect?: AnalysisRunWhereUniqueInput
    update?: XOR<XOR<AnalysisRunUpdateToOneWithWhereWithoutClusterResultsInput, AnalysisRunUpdateWithoutClusterResultsInput>, AnalysisRunUncheckedUpdateWithoutClusterResultsInput>
  }

  export type TopicAssignmentCreateNestedManyWithoutAnalysisRunInput = {
    create?: XOR<TopicAssignmentCreateWithoutAnalysisRunInput, TopicAssignmentUncheckedCreateWithoutAnalysisRunInput> | TopicAssignmentCreateWithoutAnalysisRunInput[] | TopicAssignmentUncheckedCreateWithoutAnalysisRunInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutAnalysisRunInput | TopicAssignmentCreateOrConnectWithoutAnalysisRunInput[]
    createMany?: TopicAssignmentCreateManyAnalysisRunInputEnvelope
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
  }

  export type ClusterResultCreateNestedManyWithoutAnalysisRunInput = {
    create?: XOR<ClusterResultCreateWithoutAnalysisRunInput, ClusterResultUncheckedCreateWithoutAnalysisRunInput> | ClusterResultCreateWithoutAnalysisRunInput[] | ClusterResultUncheckedCreateWithoutAnalysisRunInput[]
    connectOrCreate?: ClusterResultCreateOrConnectWithoutAnalysisRunInput | ClusterResultCreateOrConnectWithoutAnalysisRunInput[]
    createMany?: ClusterResultCreateManyAnalysisRunInputEnvelope
    connect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
  }

  export type TopicAssignmentUncheckedCreateNestedManyWithoutAnalysisRunInput = {
    create?: XOR<TopicAssignmentCreateWithoutAnalysisRunInput, TopicAssignmentUncheckedCreateWithoutAnalysisRunInput> | TopicAssignmentCreateWithoutAnalysisRunInput[] | TopicAssignmentUncheckedCreateWithoutAnalysisRunInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutAnalysisRunInput | TopicAssignmentCreateOrConnectWithoutAnalysisRunInput[]
    createMany?: TopicAssignmentCreateManyAnalysisRunInputEnvelope
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
  }

  export type ClusterResultUncheckedCreateNestedManyWithoutAnalysisRunInput = {
    create?: XOR<ClusterResultCreateWithoutAnalysisRunInput, ClusterResultUncheckedCreateWithoutAnalysisRunInput> | ClusterResultCreateWithoutAnalysisRunInput[] | ClusterResultUncheckedCreateWithoutAnalysisRunInput[]
    connectOrCreate?: ClusterResultCreateOrConnectWithoutAnalysisRunInput | ClusterResultCreateOrConnectWithoutAnalysisRunInput[]
    createMany?: ClusterResultCreateManyAnalysisRunInputEnvelope
    connect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
  }

  export type EnumAnalysisStatusFieldUpdateOperationsInput = {
    set?: $Enums.AnalysisStatus
  }

  export type NullableIntFieldUpdateOperationsInput = {
    set?: number | null
    increment?: number
    decrement?: number
    multiply?: number
    divide?: number
  }

  export type FloatFieldUpdateOperationsInput = {
    set?: number
    increment?: number
    decrement?: number
    multiply?: number
    divide?: number
  }

  export type NullableDateTimeFieldUpdateOperationsInput = {
    set?: Date | string | null
  }

  export type TopicAssignmentUpdateManyWithoutAnalysisRunNestedInput = {
    create?: XOR<TopicAssignmentCreateWithoutAnalysisRunInput, TopicAssignmentUncheckedCreateWithoutAnalysisRunInput> | TopicAssignmentCreateWithoutAnalysisRunInput[] | TopicAssignmentUncheckedCreateWithoutAnalysisRunInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutAnalysisRunInput | TopicAssignmentCreateOrConnectWithoutAnalysisRunInput[]
    upsert?: TopicAssignmentUpsertWithWhereUniqueWithoutAnalysisRunInput | TopicAssignmentUpsertWithWhereUniqueWithoutAnalysisRunInput[]
    createMany?: TopicAssignmentCreateManyAnalysisRunInputEnvelope
    set?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    disconnect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    delete?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    update?: TopicAssignmentUpdateWithWhereUniqueWithoutAnalysisRunInput | TopicAssignmentUpdateWithWhereUniqueWithoutAnalysisRunInput[]
    updateMany?: TopicAssignmentUpdateManyWithWhereWithoutAnalysisRunInput | TopicAssignmentUpdateManyWithWhereWithoutAnalysisRunInput[]
    deleteMany?: TopicAssignmentScalarWhereInput | TopicAssignmentScalarWhereInput[]
  }

  export type ClusterResultUpdateManyWithoutAnalysisRunNestedInput = {
    create?: XOR<ClusterResultCreateWithoutAnalysisRunInput, ClusterResultUncheckedCreateWithoutAnalysisRunInput> | ClusterResultCreateWithoutAnalysisRunInput[] | ClusterResultUncheckedCreateWithoutAnalysisRunInput[]
    connectOrCreate?: ClusterResultCreateOrConnectWithoutAnalysisRunInput | ClusterResultCreateOrConnectWithoutAnalysisRunInput[]
    upsert?: ClusterResultUpsertWithWhereUniqueWithoutAnalysisRunInput | ClusterResultUpsertWithWhereUniqueWithoutAnalysisRunInput[]
    createMany?: ClusterResultCreateManyAnalysisRunInputEnvelope
    set?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    disconnect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    delete?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    connect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    update?: ClusterResultUpdateWithWhereUniqueWithoutAnalysisRunInput | ClusterResultUpdateWithWhereUniqueWithoutAnalysisRunInput[]
    updateMany?: ClusterResultUpdateManyWithWhereWithoutAnalysisRunInput | ClusterResultUpdateManyWithWhereWithoutAnalysisRunInput[]
    deleteMany?: ClusterResultScalarWhereInput | ClusterResultScalarWhereInput[]
  }

  export type TopicAssignmentUncheckedUpdateManyWithoutAnalysisRunNestedInput = {
    create?: XOR<TopicAssignmentCreateWithoutAnalysisRunInput, TopicAssignmentUncheckedCreateWithoutAnalysisRunInput> | TopicAssignmentCreateWithoutAnalysisRunInput[] | TopicAssignmentUncheckedCreateWithoutAnalysisRunInput[]
    connectOrCreate?: TopicAssignmentCreateOrConnectWithoutAnalysisRunInput | TopicAssignmentCreateOrConnectWithoutAnalysisRunInput[]
    upsert?: TopicAssignmentUpsertWithWhereUniqueWithoutAnalysisRunInput | TopicAssignmentUpsertWithWhereUniqueWithoutAnalysisRunInput[]
    createMany?: TopicAssignmentCreateManyAnalysisRunInputEnvelope
    set?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    disconnect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    delete?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    connect?: TopicAssignmentWhereUniqueInput | TopicAssignmentWhereUniqueInput[]
    update?: TopicAssignmentUpdateWithWhereUniqueWithoutAnalysisRunInput | TopicAssignmentUpdateWithWhereUniqueWithoutAnalysisRunInput[]
    updateMany?: TopicAssignmentUpdateManyWithWhereWithoutAnalysisRunInput | TopicAssignmentUpdateManyWithWhereWithoutAnalysisRunInput[]
    deleteMany?: TopicAssignmentScalarWhereInput | TopicAssignmentScalarWhereInput[]
  }

  export type ClusterResultUncheckedUpdateManyWithoutAnalysisRunNestedInput = {
    create?: XOR<ClusterResultCreateWithoutAnalysisRunInput, ClusterResultUncheckedCreateWithoutAnalysisRunInput> | ClusterResultCreateWithoutAnalysisRunInput[] | ClusterResultUncheckedCreateWithoutAnalysisRunInput[]
    connectOrCreate?: ClusterResultCreateOrConnectWithoutAnalysisRunInput | ClusterResultCreateOrConnectWithoutAnalysisRunInput[]
    upsert?: ClusterResultUpsertWithWhereUniqueWithoutAnalysisRunInput | ClusterResultUpsertWithWhereUniqueWithoutAnalysisRunInput[]
    createMany?: ClusterResultCreateManyAnalysisRunInputEnvelope
    set?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    disconnect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    delete?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    connect?: ClusterResultWhereUniqueInput | ClusterResultWhereUniqueInput[]
    update?: ClusterResultUpdateWithWhereUniqueWithoutAnalysisRunInput | ClusterResultUpdateWithWhereUniqueWithoutAnalysisRunInput[]
    updateMany?: ClusterResultUpdateManyWithWhereWithoutAnalysisRunInput | ClusterResultUpdateManyWithWhereWithoutAnalysisRunInput[]
    deleteMany?: ClusterResultScalarWhereInput | ClusterResultScalarWhereInput[]
  }

  export type EnumConfigTypeFieldUpdateOperationsInput = {
    set?: $Enums.ConfigType
  }

  export type EnumSyncStatusFieldUpdateOperationsInput = {
    set?: $Enums.SyncStatus
  }

  export type NestedStringFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringFilter<$PrismaModel> | string
  }

  export type NestedStringNullableFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringNullableFilter<$PrismaModel> | string | null
  }

  export type NestedDateTimeFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeFilter<$PrismaModel> | Date | string
  }

  export type NestedStringWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringWithAggregatesFilter<$PrismaModel> | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedStringFilter<$PrismaModel>
    _max?: NestedStringFilter<$PrismaModel>
  }

  export type NestedIntFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntFilter<$PrismaModel> | number
  }

  export type NestedStringNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringNullableWithAggregatesFilter<$PrismaModel> | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedStringNullableFilter<$PrismaModel>
    _max?: NestedStringNullableFilter<$PrismaModel>
  }

  export type NestedIntNullableFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableFilter<$PrismaModel> | number | null
  }

  export type NestedDateTimeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeWithAggregatesFilter<$PrismaModel> | Date | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedDateTimeFilter<$PrismaModel>
    _max?: NestedDateTimeFilter<$PrismaModel>
  }

  export type NestedBoolFilter<$PrismaModel = never> = {
    equals?: boolean | BooleanFieldRefInput<$PrismaModel>
    not?: NestedBoolFilter<$PrismaModel> | boolean
  }

  export type NestedBoolWithAggregatesFilter<$PrismaModel = never> = {
    equals?: boolean | BooleanFieldRefInput<$PrismaModel>
    not?: NestedBoolWithAggregatesFilter<$PrismaModel> | boolean
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedBoolFilter<$PrismaModel>
    _max?: NestedBoolFilter<$PrismaModel>
  }

  export type NestedFloatNullableFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel> | null
    in?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatNullableFilter<$PrismaModel> | number | null
  }

  export type NestedEnumAssignmentTypeFilter<$PrismaModel = never> = {
    equals?: $Enums.AssignmentType | EnumAssignmentTypeFieldRefInput<$PrismaModel>
    in?: $Enums.AssignmentType[] | ListEnumAssignmentTypeFieldRefInput<$PrismaModel>
    notIn?: $Enums.AssignmentType[] | ListEnumAssignmentTypeFieldRefInput<$PrismaModel>
    not?: NestedEnumAssignmentTypeFilter<$PrismaModel> | $Enums.AssignmentType
  }

  export type NestedFloatNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel> | null
    in?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatNullableWithAggregatesFilter<$PrismaModel> | number | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _avg?: NestedFloatNullableFilter<$PrismaModel>
    _sum?: NestedFloatNullableFilter<$PrismaModel>
    _min?: NestedFloatNullableFilter<$PrismaModel>
    _max?: NestedFloatNullableFilter<$PrismaModel>
  }

  export type NestedEnumAssignmentTypeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.AssignmentType | EnumAssignmentTypeFieldRefInput<$PrismaModel>
    in?: $Enums.AssignmentType[] | ListEnumAssignmentTypeFieldRefInput<$PrismaModel>
    notIn?: $Enums.AssignmentType[] | ListEnumAssignmentTypeFieldRefInput<$PrismaModel>
    not?: NestedEnumAssignmentTypeWithAggregatesFilter<$PrismaModel> | $Enums.AssignmentType
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumAssignmentTypeFilter<$PrismaModel>
    _max?: NestedEnumAssignmentTypeFilter<$PrismaModel>
  }

  export type NestedIntWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedIntFilter<$PrismaModel>
    _min?: NestedIntFilter<$PrismaModel>
    _max?: NestedIntFilter<$PrismaModel>
  }

  export type NestedFloatFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatFilter<$PrismaModel> | number
  }

  export type NestedEnumAnalysisStatusFilter<$PrismaModel = never> = {
    equals?: $Enums.AnalysisStatus | EnumAnalysisStatusFieldRefInput<$PrismaModel>
    in?: $Enums.AnalysisStatus[] | ListEnumAnalysisStatusFieldRefInput<$PrismaModel>
    notIn?: $Enums.AnalysisStatus[] | ListEnumAnalysisStatusFieldRefInput<$PrismaModel>
    not?: NestedEnumAnalysisStatusFilter<$PrismaModel> | $Enums.AnalysisStatus
  }

  export type NestedDateTimeNullableFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel> | null
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel> | null
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel> | null
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeNullableFilter<$PrismaModel> | Date | string | null
  }

  export type NestedEnumAnalysisStatusWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.AnalysisStatus | EnumAnalysisStatusFieldRefInput<$PrismaModel>
    in?: $Enums.AnalysisStatus[] | ListEnumAnalysisStatusFieldRefInput<$PrismaModel>
    notIn?: $Enums.AnalysisStatus[] | ListEnumAnalysisStatusFieldRefInput<$PrismaModel>
    not?: NestedEnumAnalysisStatusWithAggregatesFilter<$PrismaModel> | $Enums.AnalysisStatus
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumAnalysisStatusFilter<$PrismaModel>
    _max?: NestedEnumAnalysisStatusFilter<$PrismaModel>
  }

  export type NestedIntNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableWithAggregatesFilter<$PrismaModel> | number | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _avg?: NestedFloatNullableFilter<$PrismaModel>
    _sum?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedIntNullableFilter<$PrismaModel>
    _max?: NestedIntNullableFilter<$PrismaModel>
  }

  export type NestedFloatWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedFloatFilter<$PrismaModel>
    _min?: NestedFloatFilter<$PrismaModel>
    _max?: NestedFloatFilter<$PrismaModel>
  }

  export type NestedDateTimeNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel> | null
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel> | null
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel> | null
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeNullableWithAggregatesFilter<$PrismaModel> | Date | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedDateTimeNullableFilter<$PrismaModel>
    _max?: NestedDateTimeNullableFilter<$PrismaModel>
  }
  export type NestedJsonNullableFilter<$PrismaModel = never> =
    | PatchUndefined<
        Either<Required<NestedJsonNullableFilterBase<$PrismaModel>>, Exclude<keyof Required<NestedJsonNullableFilterBase<$PrismaModel>>, 'path'>>,
        Required<NestedJsonNullableFilterBase<$PrismaModel>>
      >
    | OptionalFlat<Omit<Required<NestedJsonNullableFilterBase<$PrismaModel>>, 'path'>>

  export type NestedJsonNullableFilterBase<$PrismaModel = never> = {
    equals?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
    path?: string[]
    mode?: QueryMode | EnumQueryModeFieldRefInput<$PrismaModel>
    string_contains?: string | StringFieldRefInput<$PrismaModel>
    string_starts_with?: string | StringFieldRefInput<$PrismaModel>
    string_ends_with?: string | StringFieldRefInput<$PrismaModel>
    array_starts_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_ends_with?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    array_contains?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | null
    lt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    lte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gt?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    gte?: InputJsonValue | JsonFieldRefInput<$PrismaModel>
    not?: InputJsonValue | JsonFieldRefInput<$PrismaModel> | JsonNullValueFilter
  }

  export type NestedEnumConfigTypeFilter<$PrismaModel = never> = {
    equals?: $Enums.ConfigType | EnumConfigTypeFieldRefInput<$PrismaModel>
    in?: $Enums.ConfigType[] | ListEnumConfigTypeFieldRefInput<$PrismaModel>
    notIn?: $Enums.ConfigType[] | ListEnumConfigTypeFieldRefInput<$PrismaModel>
    not?: NestedEnumConfigTypeFilter<$PrismaModel> | $Enums.ConfigType
  }

  export type NestedEnumConfigTypeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.ConfigType | EnumConfigTypeFieldRefInput<$PrismaModel>
    in?: $Enums.ConfigType[] | ListEnumConfigTypeFieldRefInput<$PrismaModel>
    notIn?: $Enums.ConfigType[] | ListEnumConfigTypeFieldRefInput<$PrismaModel>
    not?: NestedEnumConfigTypeWithAggregatesFilter<$PrismaModel> | $Enums.ConfigType
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumConfigTypeFilter<$PrismaModel>
    _max?: NestedEnumConfigTypeFilter<$PrismaModel>
  }

  export type NestedEnumSyncStatusFilter<$PrismaModel = never> = {
    equals?: $Enums.SyncStatus | EnumSyncStatusFieldRefInput<$PrismaModel>
    in?: $Enums.SyncStatus[] | ListEnumSyncStatusFieldRefInput<$PrismaModel>
    notIn?: $Enums.SyncStatus[] | ListEnumSyncStatusFieldRefInput<$PrismaModel>
    not?: NestedEnumSyncStatusFilter<$PrismaModel> | $Enums.SyncStatus
  }

  export type NestedEnumSyncStatusWithAggregatesFilter<$PrismaModel = never> = {
    equals?: $Enums.SyncStatus | EnumSyncStatusFieldRefInput<$PrismaModel>
    in?: $Enums.SyncStatus[] | ListEnumSyncStatusFieldRefInput<$PrismaModel>
    notIn?: $Enums.SyncStatus[] | ListEnumSyncStatusFieldRefInput<$PrismaModel>
    not?: NestedEnumSyncStatusWithAggregatesFilter<$PrismaModel> | $Enums.SyncStatus
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedEnumSyncStatusFilter<$PrismaModel>
    _max?: NestedEnumSyncStatusFilter<$PrismaModel>
  }

  export type TopicAssignmentCreateWithoutQuestionInput = {
    id?: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    createdAt?: Date | string
    topic: TopicCreateNestedOneWithoutTopicAssignmentsInput
    analysisRun: AnalysisRunCreateNestedOneWithoutTopicAssignmentsInput
  }

  export type TopicAssignmentUncheckedCreateWithoutQuestionInput = {
    id?: string
    topicId: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type TopicAssignmentCreateOrConnectWithoutQuestionInput = {
    where: TopicAssignmentWhereUniqueInput
    create: XOR<TopicAssignmentCreateWithoutQuestionInput, TopicAssignmentUncheckedCreateWithoutQuestionInput>
  }

  export type TopicAssignmentCreateManyQuestionInputEnvelope = {
    data: TopicAssignmentCreateManyQuestionInput | TopicAssignmentCreateManyQuestionInput[]
    skipDuplicates?: boolean
  }

  export type ClusterResultCreateWithoutQuestionInput = {
    id?: string
    clusterId: number
    clusterName?: string | null
    createdAt?: Date | string
    analysisRun: AnalysisRunCreateNestedOneWithoutClusterResultsInput
  }

  export type ClusterResultUncheckedCreateWithoutQuestionInput = {
    id?: string
    clusterId: number
    clusterName?: string | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type ClusterResultCreateOrConnectWithoutQuestionInput = {
    where: ClusterResultWhereUniqueInput
    create: XOR<ClusterResultCreateWithoutQuestionInput, ClusterResultUncheckedCreateWithoutQuestionInput>
  }

  export type ClusterResultCreateManyQuestionInputEnvelope = {
    data: ClusterResultCreateManyQuestionInput | ClusterResultCreateManyQuestionInput[]
    skipDuplicates?: boolean
  }

  export type TopicAssignmentUpsertWithWhereUniqueWithoutQuestionInput = {
    where: TopicAssignmentWhereUniqueInput
    update: XOR<TopicAssignmentUpdateWithoutQuestionInput, TopicAssignmentUncheckedUpdateWithoutQuestionInput>
    create: XOR<TopicAssignmentCreateWithoutQuestionInput, TopicAssignmentUncheckedCreateWithoutQuestionInput>
  }

  export type TopicAssignmentUpdateWithWhereUniqueWithoutQuestionInput = {
    where: TopicAssignmentWhereUniqueInput
    data: XOR<TopicAssignmentUpdateWithoutQuestionInput, TopicAssignmentUncheckedUpdateWithoutQuestionInput>
  }

  export type TopicAssignmentUpdateManyWithWhereWithoutQuestionInput = {
    where: TopicAssignmentScalarWhereInput
    data: XOR<TopicAssignmentUpdateManyMutationInput, TopicAssignmentUncheckedUpdateManyWithoutQuestionInput>
  }

  export type TopicAssignmentScalarWhereInput = {
    AND?: TopicAssignmentScalarWhereInput | TopicAssignmentScalarWhereInput[]
    OR?: TopicAssignmentScalarWhereInput[]
    NOT?: TopicAssignmentScalarWhereInput | TopicAssignmentScalarWhereInput[]
    id?: StringFilter<"TopicAssignment"> | string
    questionId?: StringFilter<"TopicAssignment"> | string
    topicId?: StringFilter<"TopicAssignment"> | string
    similarityScore?: FloatNullableFilter<"TopicAssignment"> | number | null
    assignmentType?: EnumAssignmentTypeFilter<"TopicAssignment"> | $Enums.AssignmentType
    confidence?: FloatNullableFilter<"TopicAssignment"> | number | null
    analysisRunId?: StringFilter<"TopicAssignment"> | string
    createdAt?: DateTimeFilter<"TopicAssignment"> | Date | string
  }

  export type ClusterResultUpsertWithWhereUniqueWithoutQuestionInput = {
    where: ClusterResultWhereUniqueInput
    update: XOR<ClusterResultUpdateWithoutQuestionInput, ClusterResultUncheckedUpdateWithoutQuestionInput>
    create: XOR<ClusterResultCreateWithoutQuestionInput, ClusterResultUncheckedCreateWithoutQuestionInput>
  }

  export type ClusterResultUpdateWithWhereUniqueWithoutQuestionInput = {
    where: ClusterResultWhereUniqueInput
    data: XOR<ClusterResultUpdateWithoutQuestionInput, ClusterResultUncheckedUpdateWithoutQuestionInput>
  }

  export type ClusterResultUpdateManyWithWhereWithoutQuestionInput = {
    where: ClusterResultScalarWhereInput
    data: XOR<ClusterResultUpdateManyMutationInput, ClusterResultUncheckedUpdateManyWithoutQuestionInput>
  }

  export type ClusterResultScalarWhereInput = {
    AND?: ClusterResultScalarWhereInput | ClusterResultScalarWhereInput[]
    OR?: ClusterResultScalarWhereInput[]
    NOT?: ClusterResultScalarWhereInput | ClusterResultScalarWhereInput[]
    id?: StringFilter<"ClusterResult"> | string
    questionId?: StringFilter<"ClusterResult"> | string
    clusterId?: IntFilter<"ClusterResult"> | number
    clusterName?: StringNullableFilter<"ClusterResult"> | string | null
    analysisRunId?: StringFilter<"ClusterResult"> | string
    createdAt?: DateTimeFilter<"ClusterResult"> | Date | string
  }

  export type TopicAssignmentCreateWithoutTopicInput = {
    id?: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    createdAt?: Date | string
    question: QuestionCreateNestedOneWithoutTopicAssignmentsInput
    analysisRun: AnalysisRunCreateNestedOneWithoutTopicAssignmentsInput
  }

  export type TopicAssignmentUncheckedCreateWithoutTopicInput = {
    id?: string
    questionId: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type TopicAssignmentCreateOrConnectWithoutTopicInput = {
    where: TopicAssignmentWhereUniqueInput
    create: XOR<TopicAssignmentCreateWithoutTopicInput, TopicAssignmentUncheckedCreateWithoutTopicInput>
  }

  export type TopicAssignmentCreateManyTopicInputEnvelope = {
    data: TopicAssignmentCreateManyTopicInput | TopicAssignmentCreateManyTopicInput[]
    skipDuplicates?: boolean
  }

  export type TopicAssignmentUpsertWithWhereUniqueWithoutTopicInput = {
    where: TopicAssignmentWhereUniqueInput
    update: XOR<TopicAssignmentUpdateWithoutTopicInput, TopicAssignmentUncheckedUpdateWithoutTopicInput>
    create: XOR<TopicAssignmentCreateWithoutTopicInput, TopicAssignmentUncheckedCreateWithoutTopicInput>
  }

  export type TopicAssignmentUpdateWithWhereUniqueWithoutTopicInput = {
    where: TopicAssignmentWhereUniqueInput
    data: XOR<TopicAssignmentUpdateWithoutTopicInput, TopicAssignmentUncheckedUpdateWithoutTopicInput>
  }

  export type TopicAssignmentUpdateManyWithWhereWithoutTopicInput = {
    where: TopicAssignmentScalarWhereInput
    data: XOR<TopicAssignmentUpdateManyMutationInput, TopicAssignmentUncheckedUpdateManyWithoutTopicInput>
  }

  export type QuestionCreateWithoutTopicAssignmentsInput = {
    id?: string
    text: string
    originalText?: string | null
    embedding?: QuestionCreateembeddingInput | number[]
    language?: string
    country?: string | null
    state?: string | null
    userId?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
    clusterResults?: ClusterResultCreateNestedManyWithoutQuestionInput
  }

  export type QuestionUncheckedCreateWithoutTopicAssignmentsInput = {
    id?: string
    text: string
    originalText?: string | null
    embedding?: QuestionCreateembeddingInput | number[]
    language?: string
    country?: string | null
    state?: string | null
    userId?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
    clusterResults?: ClusterResultUncheckedCreateNestedManyWithoutQuestionInput
  }

  export type QuestionCreateOrConnectWithoutTopicAssignmentsInput = {
    where: QuestionWhereUniqueInput
    create: XOR<QuestionCreateWithoutTopicAssignmentsInput, QuestionUncheckedCreateWithoutTopicAssignmentsInput>
  }

  export type TopicCreateWithoutTopicAssignmentsInput = {
    id?: string
    name: string
    description?: string | null
    subtopic?: string | null
    isSystemTopic?: boolean
    isActive?: boolean
    embedding?: TopicCreateembeddingInput | number[]
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type TopicUncheckedCreateWithoutTopicAssignmentsInput = {
    id?: string
    name: string
    description?: string | null
    subtopic?: string | null
    isSystemTopic?: boolean
    isActive?: boolean
    embedding?: TopicCreateembeddingInput | number[]
    createdAt?: Date | string
    updatedAt?: Date | string
  }

  export type TopicCreateOrConnectWithoutTopicAssignmentsInput = {
    where: TopicWhereUniqueInput
    create: XOR<TopicCreateWithoutTopicAssignmentsInput, TopicUncheckedCreateWithoutTopicAssignmentsInput>
  }

  export type AnalysisRunCreateWithoutTopicAssignmentsInput = {
    id?: string
    status?: $Enums.AnalysisStatus
    mode: string
    sampleSize?: number | null
    similarityThreshold: number
    embeddingModel: string
    gptModel: string
    totalQuestions?: number | null
    matchedQuestions?: number | null
    newTopicsFound?: number | null
    processingTimeMs?: number | null
    errorMessage?: string | null
    startedAt?: Date | string
    completedAt?: Date | string | null
    createdBy?: string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    clusterResults?: ClusterResultCreateNestedManyWithoutAnalysisRunInput
  }

  export type AnalysisRunUncheckedCreateWithoutTopicAssignmentsInput = {
    id?: string
    status?: $Enums.AnalysisStatus
    mode: string
    sampleSize?: number | null
    similarityThreshold: number
    embeddingModel: string
    gptModel: string
    totalQuestions?: number | null
    matchedQuestions?: number | null
    newTopicsFound?: number | null
    processingTimeMs?: number | null
    errorMessage?: string | null
    startedAt?: Date | string
    completedAt?: Date | string | null
    createdBy?: string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    clusterResults?: ClusterResultUncheckedCreateNestedManyWithoutAnalysisRunInput
  }

  export type AnalysisRunCreateOrConnectWithoutTopicAssignmentsInput = {
    where: AnalysisRunWhereUniqueInput
    create: XOR<AnalysisRunCreateWithoutTopicAssignmentsInput, AnalysisRunUncheckedCreateWithoutTopicAssignmentsInput>
  }

  export type QuestionUpsertWithoutTopicAssignmentsInput = {
    update: XOR<QuestionUpdateWithoutTopicAssignmentsInput, QuestionUncheckedUpdateWithoutTopicAssignmentsInput>
    create: XOR<QuestionCreateWithoutTopicAssignmentsInput, QuestionUncheckedCreateWithoutTopicAssignmentsInput>
    where?: QuestionWhereInput
  }

  export type QuestionUpdateToOneWithWhereWithoutTopicAssignmentsInput = {
    where?: QuestionWhereInput
    data: XOR<QuestionUpdateWithoutTopicAssignmentsInput, QuestionUncheckedUpdateWithoutTopicAssignmentsInput>
  }

  export type QuestionUpdateWithoutTopicAssignmentsInput = {
    id?: StringFieldUpdateOperationsInput | string
    text?: StringFieldUpdateOperationsInput | string
    originalText?: NullableStringFieldUpdateOperationsInput | string | null
    embedding?: QuestionUpdateembeddingInput | number[]
    language?: StringFieldUpdateOperationsInput | string
    country?: NullableStringFieldUpdateOperationsInput | string | null
    state?: NullableStringFieldUpdateOperationsInput | string | null
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    clusterResults?: ClusterResultUpdateManyWithoutQuestionNestedInput
  }

  export type QuestionUncheckedUpdateWithoutTopicAssignmentsInput = {
    id?: StringFieldUpdateOperationsInput | string
    text?: StringFieldUpdateOperationsInput | string
    originalText?: NullableStringFieldUpdateOperationsInput | string | null
    embedding?: QuestionUpdateembeddingInput | number[]
    language?: StringFieldUpdateOperationsInput | string
    country?: NullableStringFieldUpdateOperationsInput | string | null
    state?: NullableStringFieldUpdateOperationsInput | string | null
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    clusterResults?: ClusterResultUncheckedUpdateManyWithoutQuestionNestedInput
  }

  export type TopicUpsertWithoutTopicAssignmentsInput = {
    update: XOR<TopicUpdateWithoutTopicAssignmentsInput, TopicUncheckedUpdateWithoutTopicAssignmentsInput>
    create: XOR<TopicCreateWithoutTopicAssignmentsInput, TopicUncheckedCreateWithoutTopicAssignmentsInput>
    where?: TopicWhereInput
  }

  export type TopicUpdateToOneWithWhereWithoutTopicAssignmentsInput = {
    where?: TopicWhereInput
    data: XOR<TopicUpdateWithoutTopicAssignmentsInput, TopicUncheckedUpdateWithoutTopicAssignmentsInput>
  }

  export type TopicUpdateWithoutTopicAssignmentsInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: NullableStringFieldUpdateOperationsInput | string | null
    subtopic?: NullableStringFieldUpdateOperationsInput | string | null
    isSystemTopic?: BoolFieldUpdateOperationsInput | boolean
    isActive?: BoolFieldUpdateOperationsInput | boolean
    embedding?: TopicUpdateembeddingInput | number[]
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicUncheckedUpdateWithoutTopicAssignmentsInput = {
    id?: StringFieldUpdateOperationsInput | string
    name?: StringFieldUpdateOperationsInput | string
    description?: NullableStringFieldUpdateOperationsInput | string | null
    subtopic?: NullableStringFieldUpdateOperationsInput | string | null
    isSystemTopic?: BoolFieldUpdateOperationsInput | boolean
    isActive?: BoolFieldUpdateOperationsInput | boolean
    embedding?: TopicUpdateembeddingInput | number[]
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type AnalysisRunUpsertWithoutTopicAssignmentsInput = {
    update: XOR<AnalysisRunUpdateWithoutTopicAssignmentsInput, AnalysisRunUncheckedUpdateWithoutTopicAssignmentsInput>
    create: XOR<AnalysisRunCreateWithoutTopicAssignmentsInput, AnalysisRunUncheckedCreateWithoutTopicAssignmentsInput>
    where?: AnalysisRunWhereInput
  }

  export type AnalysisRunUpdateToOneWithWhereWithoutTopicAssignmentsInput = {
    where?: AnalysisRunWhereInput
    data: XOR<AnalysisRunUpdateWithoutTopicAssignmentsInput, AnalysisRunUncheckedUpdateWithoutTopicAssignmentsInput>
  }

  export type AnalysisRunUpdateWithoutTopicAssignmentsInput = {
    id?: StringFieldUpdateOperationsInput | string
    status?: EnumAnalysisStatusFieldUpdateOperationsInput | $Enums.AnalysisStatus
    mode?: StringFieldUpdateOperationsInput | string
    sampleSize?: NullableIntFieldUpdateOperationsInput | number | null
    similarityThreshold?: FloatFieldUpdateOperationsInput | number
    embeddingModel?: StringFieldUpdateOperationsInput | string
    gptModel?: StringFieldUpdateOperationsInput | string
    totalQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    matchedQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    newTopicsFound?: NullableIntFieldUpdateOperationsInput | number | null
    processingTimeMs?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
    startedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    completedAt?: NullableDateTimeFieldUpdateOperationsInput | Date | string | null
    createdBy?: NullableStringFieldUpdateOperationsInput | string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    clusterResults?: ClusterResultUpdateManyWithoutAnalysisRunNestedInput
  }

  export type AnalysisRunUncheckedUpdateWithoutTopicAssignmentsInput = {
    id?: StringFieldUpdateOperationsInput | string
    status?: EnumAnalysisStatusFieldUpdateOperationsInput | $Enums.AnalysisStatus
    mode?: StringFieldUpdateOperationsInput | string
    sampleSize?: NullableIntFieldUpdateOperationsInput | number | null
    similarityThreshold?: FloatFieldUpdateOperationsInput | number
    embeddingModel?: StringFieldUpdateOperationsInput | string
    gptModel?: StringFieldUpdateOperationsInput | string
    totalQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    matchedQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    newTopicsFound?: NullableIntFieldUpdateOperationsInput | number | null
    processingTimeMs?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
    startedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    completedAt?: NullableDateTimeFieldUpdateOperationsInput | Date | string | null
    createdBy?: NullableStringFieldUpdateOperationsInput | string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    clusterResults?: ClusterResultUncheckedUpdateManyWithoutAnalysisRunNestedInput
  }

  export type QuestionCreateWithoutClusterResultsInput = {
    id?: string
    text: string
    originalText?: string | null
    embedding?: QuestionCreateembeddingInput | number[]
    language?: string
    country?: string | null
    state?: string | null
    userId?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
    topicAssignments?: TopicAssignmentCreateNestedManyWithoutQuestionInput
  }

  export type QuestionUncheckedCreateWithoutClusterResultsInput = {
    id?: string
    text: string
    originalText?: string | null
    embedding?: QuestionCreateembeddingInput | number[]
    language?: string
    country?: string | null
    state?: string | null
    userId?: string | null
    createdAt?: Date | string
    updatedAt?: Date | string
    topicAssignments?: TopicAssignmentUncheckedCreateNestedManyWithoutQuestionInput
  }

  export type QuestionCreateOrConnectWithoutClusterResultsInput = {
    where: QuestionWhereUniqueInput
    create: XOR<QuestionCreateWithoutClusterResultsInput, QuestionUncheckedCreateWithoutClusterResultsInput>
  }

  export type AnalysisRunCreateWithoutClusterResultsInput = {
    id?: string
    status?: $Enums.AnalysisStatus
    mode: string
    sampleSize?: number | null
    similarityThreshold: number
    embeddingModel: string
    gptModel: string
    totalQuestions?: number | null
    matchedQuestions?: number | null
    newTopicsFound?: number | null
    processingTimeMs?: number | null
    errorMessage?: string | null
    startedAt?: Date | string
    completedAt?: Date | string | null
    createdBy?: string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    topicAssignments?: TopicAssignmentCreateNestedManyWithoutAnalysisRunInput
  }

  export type AnalysisRunUncheckedCreateWithoutClusterResultsInput = {
    id?: string
    status?: $Enums.AnalysisStatus
    mode: string
    sampleSize?: number | null
    similarityThreshold: number
    embeddingModel: string
    gptModel: string
    totalQuestions?: number | null
    matchedQuestions?: number | null
    newTopicsFound?: number | null
    processingTimeMs?: number | null
    errorMessage?: string | null
    startedAt?: Date | string
    completedAt?: Date | string | null
    createdBy?: string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    topicAssignments?: TopicAssignmentUncheckedCreateNestedManyWithoutAnalysisRunInput
  }

  export type AnalysisRunCreateOrConnectWithoutClusterResultsInput = {
    where: AnalysisRunWhereUniqueInput
    create: XOR<AnalysisRunCreateWithoutClusterResultsInput, AnalysisRunUncheckedCreateWithoutClusterResultsInput>
  }

  export type QuestionUpsertWithoutClusterResultsInput = {
    update: XOR<QuestionUpdateWithoutClusterResultsInput, QuestionUncheckedUpdateWithoutClusterResultsInput>
    create: XOR<QuestionCreateWithoutClusterResultsInput, QuestionUncheckedCreateWithoutClusterResultsInput>
    where?: QuestionWhereInput
  }

  export type QuestionUpdateToOneWithWhereWithoutClusterResultsInput = {
    where?: QuestionWhereInput
    data: XOR<QuestionUpdateWithoutClusterResultsInput, QuestionUncheckedUpdateWithoutClusterResultsInput>
  }

  export type QuestionUpdateWithoutClusterResultsInput = {
    id?: StringFieldUpdateOperationsInput | string
    text?: StringFieldUpdateOperationsInput | string
    originalText?: NullableStringFieldUpdateOperationsInput | string | null
    embedding?: QuestionUpdateembeddingInput | number[]
    language?: StringFieldUpdateOperationsInput | string
    country?: NullableStringFieldUpdateOperationsInput | string | null
    state?: NullableStringFieldUpdateOperationsInput | string | null
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    topicAssignments?: TopicAssignmentUpdateManyWithoutQuestionNestedInput
  }

  export type QuestionUncheckedUpdateWithoutClusterResultsInput = {
    id?: StringFieldUpdateOperationsInput | string
    text?: StringFieldUpdateOperationsInput | string
    originalText?: NullableStringFieldUpdateOperationsInput | string | null
    embedding?: QuestionUpdateembeddingInput | number[]
    language?: StringFieldUpdateOperationsInput | string
    country?: NullableStringFieldUpdateOperationsInput | string | null
    state?: NullableStringFieldUpdateOperationsInput | string | null
    userId?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    updatedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    topicAssignments?: TopicAssignmentUncheckedUpdateManyWithoutQuestionNestedInput
  }

  export type AnalysisRunUpsertWithoutClusterResultsInput = {
    update: XOR<AnalysisRunUpdateWithoutClusterResultsInput, AnalysisRunUncheckedUpdateWithoutClusterResultsInput>
    create: XOR<AnalysisRunCreateWithoutClusterResultsInput, AnalysisRunUncheckedCreateWithoutClusterResultsInput>
    where?: AnalysisRunWhereInput
  }

  export type AnalysisRunUpdateToOneWithWhereWithoutClusterResultsInput = {
    where?: AnalysisRunWhereInput
    data: XOR<AnalysisRunUpdateWithoutClusterResultsInput, AnalysisRunUncheckedUpdateWithoutClusterResultsInput>
  }

  export type AnalysisRunUpdateWithoutClusterResultsInput = {
    id?: StringFieldUpdateOperationsInput | string
    status?: EnumAnalysisStatusFieldUpdateOperationsInput | $Enums.AnalysisStatus
    mode?: StringFieldUpdateOperationsInput | string
    sampleSize?: NullableIntFieldUpdateOperationsInput | number | null
    similarityThreshold?: FloatFieldUpdateOperationsInput | number
    embeddingModel?: StringFieldUpdateOperationsInput | string
    gptModel?: StringFieldUpdateOperationsInput | string
    totalQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    matchedQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    newTopicsFound?: NullableIntFieldUpdateOperationsInput | number | null
    processingTimeMs?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
    startedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    completedAt?: NullableDateTimeFieldUpdateOperationsInput | Date | string | null
    createdBy?: NullableStringFieldUpdateOperationsInput | string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    topicAssignments?: TopicAssignmentUpdateManyWithoutAnalysisRunNestedInput
  }

  export type AnalysisRunUncheckedUpdateWithoutClusterResultsInput = {
    id?: StringFieldUpdateOperationsInput | string
    status?: EnumAnalysisStatusFieldUpdateOperationsInput | $Enums.AnalysisStatus
    mode?: StringFieldUpdateOperationsInput | string
    sampleSize?: NullableIntFieldUpdateOperationsInput | number | null
    similarityThreshold?: FloatFieldUpdateOperationsInput | number
    embeddingModel?: StringFieldUpdateOperationsInput | string
    gptModel?: StringFieldUpdateOperationsInput | string
    totalQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    matchedQuestions?: NullableIntFieldUpdateOperationsInput | number | null
    newTopicsFound?: NullableIntFieldUpdateOperationsInput | number | null
    processingTimeMs?: NullableIntFieldUpdateOperationsInput | number | null
    errorMessage?: NullableStringFieldUpdateOperationsInput | string | null
    startedAt?: DateTimeFieldUpdateOperationsInput | Date | string
    completedAt?: NullableDateTimeFieldUpdateOperationsInput | Date | string | null
    createdBy?: NullableStringFieldUpdateOperationsInput | string | null
    config?: NullableJsonNullValueInput | InputJsonValue
    topicAssignments?: TopicAssignmentUncheckedUpdateManyWithoutAnalysisRunNestedInput
  }

  export type TopicAssignmentCreateWithoutAnalysisRunInput = {
    id?: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    createdAt?: Date | string
    question: QuestionCreateNestedOneWithoutTopicAssignmentsInput
    topic: TopicCreateNestedOneWithoutTopicAssignmentsInput
  }

  export type TopicAssignmentUncheckedCreateWithoutAnalysisRunInput = {
    id?: string
    questionId: string
    topicId: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    createdAt?: Date | string
  }

  export type TopicAssignmentCreateOrConnectWithoutAnalysisRunInput = {
    where: TopicAssignmentWhereUniqueInput
    create: XOR<TopicAssignmentCreateWithoutAnalysisRunInput, TopicAssignmentUncheckedCreateWithoutAnalysisRunInput>
  }

  export type TopicAssignmentCreateManyAnalysisRunInputEnvelope = {
    data: TopicAssignmentCreateManyAnalysisRunInput | TopicAssignmentCreateManyAnalysisRunInput[]
    skipDuplicates?: boolean
  }

  export type ClusterResultCreateWithoutAnalysisRunInput = {
    id?: string
    clusterId: number
    clusterName?: string | null
    createdAt?: Date | string
    question: QuestionCreateNestedOneWithoutClusterResultsInput
  }

  export type ClusterResultUncheckedCreateWithoutAnalysisRunInput = {
    id?: string
    questionId: string
    clusterId: number
    clusterName?: string | null
    createdAt?: Date | string
  }

  export type ClusterResultCreateOrConnectWithoutAnalysisRunInput = {
    where: ClusterResultWhereUniqueInput
    create: XOR<ClusterResultCreateWithoutAnalysisRunInput, ClusterResultUncheckedCreateWithoutAnalysisRunInput>
  }

  export type ClusterResultCreateManyAnalysisRunInputEnvelope = {
    data: ClusterResultCreateManyAnalysisRunInput | ClusterResultCreateManyAnalysisRunInput[]
    skipDuplicates?: boolean
  }

  export type TopicAssignmentUpsertWithWhereUniqueWithoutAnalysisRunInput = {
    where: TopicAssignmentWhereUniqueInput
    update: XOR<TopicAssignmentUpdateWithoutAnalysisRunInput, TopicAssignmentUncheckedUpdateWithoutAnalysisRunInput>
    create: XOR<TopicAssignmentCreateWithoutAnalysisRunInput, TopicAssignmentUncheckedCreateWithoutAnalysisRunInput>
  }

  export type TopicAssignmentUpdateWithWhereUniqueWithoutAnalysisRunInput = {
    where: TopicAssignmentWhereUniqueInput
    data: XOR<TopicAssignmentUpdateWithoutAnalysisRunInput, TopicAssignmentUncheckedUpdateWithoutAnalysisRunInput>
  }

  export type TopicAssignmentUpdateManyWithWhereWithoutAnalysisRunInput = {
    where: TopicAssignmentScalarWhereInput
    data: XOR<TopicAssignmentUpdateManyMutationInput, TopicAssignmentUncheckedUpdateManyWithoutAnalysisRunInput>
  }

  export type ClusterResultUpsertWithWhereUniqueWithoutAnalysisRunInput = {
    where: ClusterResultWhereUniqueInput
    update: XOR<ClusterResultUpdateWithoutAnalysisRunInput, ClusterResultUncheckedUpdateWithoutAnalysisRunInput>
    create: XOR<ClusterResultCreateWithoutAnalysisRunInput, ClusterResultUncheckedCreateWithoutAnalysisRunInput>
  }

  export type ClusterResultUpdateWithWhereUniqueWithoutAnalysisRunInput = {
    where: ClusterResultWhereUniqueInput
    data: XOR<ClusterResultUpdateWithoutAnalysisRunInput, ClusterResultUncheckedUpdateWithoutAnalysisRunInput>
  }

  export type ClusterResultUpdateManyWithWhereWithoutAnalysisRunInput = {
    where: ClusterResultScalarWhereInput
    data: XOR<ClusterResultUpdateManyMutationInput, ClusterResultUncheckedUpdateManyWithoutAnalysisRunInput>
  }

  export type TopicAssignmentCreateManyQuestionInput = {
    id?: string
    topicId: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type ClusterResultCreateManyQuestionInput = {
    id?: string
    clusterId: number
    clusterName?: string | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type TopicAssignmentUpdateWithoutQuestionInput = {
    id?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    topic?: TopicUpdateOneRequiredWithoutTopicAssignmentsNestedInput
    analysisRun?: AnalysisRunUpdateOneRequiredWithoutTopicAssignmentsNestedInput
  }

  export type TopicAssignmentUncheckedUpdateWithoutQuestionInput = {
    id?: StringFieldUpdateOperationsInput | string
    topicId?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicAssignmentUncheckedUpdateManyWithoutQuestionInput = {
    id?: StringFieldUpdateOperationsInput | string
    topicId?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ClusterResultUpdateWithoutQuestionInput = {
    id?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    analysisRun?: AnalysisRunUpdateOneRequiredWithoutClusterResultsNestedInput
  }

  export type ClusterResultUncheckedUpdateWithoutQuestionInput = {
    id?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ClusterResultUncheckedUpdateManyWithoutQuestionInput = {
    id?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicAssignmentCreateManyTopicInput = {
    id?: string
    questionId: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    analysisRunId: string
    createdAt?: Date | string
  }

  export type TopicAssignmentUpdateWithoutTopicInput = {
    id?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    question?: QuestionUpdateOneRequiredWithoutTopicAssignmentsNestedInput
    analysisRun?: AnalysisRunUpdateOneRequiredWithoutTopicAssignmentsNestedInput
  }

  export type TopicAssignmentUncheckedUpdateWithoutTopicInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicAssignmentUncheckedUpdateManyWithoutTopicInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    analysisRunId?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicAssignmentCreateManyAnalysisRunInput = {
    id?: string
    questionId: string
    topicId: string
    similarityScore?: number | null
    assignmentType: $Enums.AssignmentType
    confidence?: number | null
    createdAt?: Date | string
  }

  export type ClusterResultCreateManyAnalysisRunInput = {
    id?: string
    questionId: string
    clusterId: number
    clusterName?: string | null
    createdAt?: Date | string
  }

  export type TopicAssignmentUpdateWithoutAnalysisRunInput = {
    id?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    question?: QuestionUpdateOneRequiredWithoutTopicAssignmentsNestedInput
    topic?: TopicUpdateOneRequiredWithoutTopicAssignmentsNestedInput
  }

  export type TopicAssignmentUncheckedUpdateWithoutAnalysisRunInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    topicId?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type TopicAssignmentUncheckedUpdateManyWithoutAnalysisRunInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    topicId?: StringFieldUpdateOperationsInput | string
    similarityScore?: NullableFloatFieldUpdateOperationsInput | number | null
    assignmentType?: EnumAssignmentTypeFieldUpdateOperationsInput | $Enums.AssignmentType
    confidence?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ClusterResultUpdateWithoutAnalysisRunInput = {
    id?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    question?: QuestionUpdateOneRequiredWithoutClusterResultsNestedInput
  }

  export type ClusterResultUncheckedUpdateWithoutAnalysisRunInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ClusterResultUncheckedUpdateManyWithoutAnalysisRunInput = {
    id?: StringFieldUpdateOperationsInput | string
    questionId?: StringFieldUpdateOperationsInput | string
    clusterId?: IntFieldUpdateOperationsInput | number
    clusterName?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }



  /**
   * Batch Payload for updateMany & deleteMany & createMany
   */

  export type BatchPayload = {
    count: number
  }

  /**
   * DMMF
   */
  export const dmmf: runtime.BaseDMMF
}